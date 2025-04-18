import torch
import torch.amp as amp
from torch.backends.cuda import sdp_kernel
from xfuser.core.distributed import get_sequence_parallel_rank
from xfuser.core.distributed import get_sequence_parallel_world_size
from xfuser.core.distributed import get_sp_group
from xfuser.core.long_ctx_attention import xFuserLongContextAttention

from ..modules.transformer import sinusoidal_embedding_1d


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(pad_size, s1, s2, dtype=original_tensor.dtype, device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    grid = [grid_sizes.tolist()] * x.size(0)
    for i, (f, h, w) in enumerate(grid):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank) : ((sp_rank + 1) * s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank.cuda()).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def usp_dit_forward(self, x, t, context, clip_fea=None, y=None, fps=None):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == "i2v":
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = torch.cat([x, y], dim=1)

    # embeddings
    x = self.patch_embedding(x)
    grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)
    x = x.flatten(2).transpose(1, 2)

    if self.flag_causal_attention:
        frame_num = grid_sizes[0]
        height = grid_sizes[1]
        width = grid_sizes[2]
        block_num = frame_num // self.num_frame_per_block
        range_tensor = torch.arange(block_num).view(-1, 1)
        range_tensor = range_tensor.repeat(1, self.num_frame_per_block).flatten()
        casual_mask = range_tensor.unsqueeze(0) <= range_tensor.unsqueeze(1)  # f, f
        casual_mask = casual_mask.view(frame_num, 1, 1, frame_num, 1, 1).to(x.device)
        casual_mask = casual_mask.repeat(1, height, width, 1, height, width)
        casual_mask = casual_mask.reshape(frame_num * height * width, frame_num * height * width)
        self.block_mask = casual_mask.unsqueeze(0).unsqueeze(0)

    # time embeddings
    with amp.autocast("cuda", dtype=torch.float32):
        if t.dim() == 2:
            b, f = t.shape
            _flag_df = True
        else:
            _flag_df = False
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(self.patch_embedding.weight.dtype)
        )  # b, dim
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # b, 6, dim

        if self.inject_sample_info:
            fps = torch.tensor(fps, dtype=torch.long, device=device)

            fps_emb = self.fps_embedding(fps).float()
            if _flag_df:
                e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim)).repeat(t.shape[1], 1, 1)
            else:
                e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim))

        if _flag_df:
            e = e.view(b, f, 1, 1, self.dim)
            e0 = e0.view(b, f, 1, 1, 6, self.dim)
            e = e.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1).flatten(1, 3)
            e0 = e0.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1, 1).flatten(1, 3)
            e0 = e0.transpose(1, 2).contiguous()

        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context = self.text_embedding(context)

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    if e0.ndim == 4:
        e0 = torch.chunk(e0, get_sequence_parallel_world_size(), dim=2)[get_sequence_parallel_rank()]
    kwargs = dict(e=e0, grid_sizes=grid_sizes, freqs=self.freqs, context=context, block_mask=self.block_mask)

    # Context Parallel
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    if e.ndim == 3:
        e = torch.chunk(e, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    x = self.head(x, e)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x.float()


def usp_attn_forward(self, x, grid_sizes, freqs, block_mask):

    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        seq_lens(Tensor): Shape [B]
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    x = x.to(self.q.weight.dtype)
    q, k, v = qkv_fn(x)

    if not self._flag_ar_attention:
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
    else:

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        # x = torch.nn.functional.scaled_dot_product_attention(
        #     q.transpose(1, 2),
        #     k.transpose(1, 2),
        #     v.transpose(1, 2),
        #    ).transpose(1, 2).contiguous()
        with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            x = (
                torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=block_mask
                )
                .transpose(1, 2)
                .contiguous()
            )
    x = xFuserLongContextAttention()(None, query=half(q), key=half(k), value=half(v), window_size=self.window_size)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
