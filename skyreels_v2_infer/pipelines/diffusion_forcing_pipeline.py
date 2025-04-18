import math
import os
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm

from ..modules import get_text_encoder
from ..modules import get_transformer
from ..modules import get_vae
from ..scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler


class DiffusionForcingPipeline:
    """
    A pipeline for diffusion-based video generation tasks.

    This pipeline supports two main tasks:
    - Image-to-Video (i2v): Generates a video sequence from a source image
    - Text-to-Video (t2v): Generates a video sequence from a text description

    The pipeline integrates multiple components including:
    - A transformer model for diffusion
    - A VAE for encoding/decoding
    - A text encoder for processing text prompts
    - An image encoder for processing image inputs (i2v mode only)
    """

    def __init__(
        self,
        model_path: str,
        dit_path: str,
        device: str = "cuda",
        weight_dtype=torch.bfloat16,
        use_usp=False,
        offload=False,
    ):
        """
        Initialize the diffusion forcing pipeline class

        Args:
            model_path (str): Path to the model
            dit_path (str): Path to the DIT model, containing model configuration file (config.json) and weight file (*.safetensor)
            device (str): Device to run on, defaults to 'cuda'
            weight_dtype: Weight data type, defaults to torch.bfloat16
        """
        load_device = "cpu" if offload else device
        self.transformer = get_transformer(dit_path, load_device, weight_dtype)
        vae_model_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        self.vae = get_vae(vae_model_path, device, weight_dtype=torch.float32)
        self.text_encoder = get_text_encoder(model_path, load_device, weight_dtype)
        self.video_processor = VideoProcessor(vae_scale_factor=16)
        self.device = device
        self.offload = offload

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward
            import types

            for block in self.transformer.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
                self.transformer.forward = types.MethodType(usp_dit_forward, self.transformer)
                self.sp_size = get_sequence_parallel_world_size()

        self.scheduler = FlowUniPCMultistepScheduler()

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1

    def encode_image(
        self, image: PipelineImageInput, height: int, width: int, num_frames: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # prefix_video
        prefix_video = np.array(image.resize((width, height))).transpose(2, 0, 1)
        prefix_video = torch.tensor(prefix_video).unsqueeze(1)  # .to(image_embeds.dtype).unsqueeze(1)
        if prefix_video.dtype == torch.uint8:
            prefix_video = (prefix_video.float() / (255.0 / 2.0)) - 1.0
        prefix_video = prefix_video.to(self.device)
        prefix_video = [self.vae.encode(prefix_video.unsqueeze(0))[0]]  # [(c, f, h, w)]
        causal_block_size = self.transformer.num_frame_per_block
        if prefix_video[0].shape[1] % causal_block_size != 0:
            truncate_len = prefix_video[0].shape[1] % causal_block_size
            print("the length of prefix video is truncated for the casual block size alignment.")
            prefix_video[0] = prefix_video[0][:, : prefix_video[0].shape[1] - truncate_len]
        predix_video_latent_length = prefix_video[0].shape[1]
        return prefix_video, predix_video_latent_length

    def prepare_latents(
        self,
        shape: Tuple[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        return randn_tensor(shape, generator, device=device, dtype=dtype)

    def generate_timestep_matrix(
        self,
        num_frames,
        step_template,
        base_num_frames,
        ar_step=5,
        num_pre_ready=0,
        casual_block_size=1,
        shrink_interval_with_mask=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple]]:
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template) + 1
        num_frames_block = num_frames // casual_block_size
        base_num_frames_block = base_num_frames // casual_block_size
        if base_num_frames_block < num_frames_block:
            infer_step_num = len(step_template)
            gen_block = base_num_frames_block
            min_ar_step = infer_step_num / gen_block
            assert ar_step >= min_ar_step, f"ar_step should be at least {math.ceil(min_ar_step)} in your setting"
        # print(num_frames, step_template, base_num_frames, ar_step, num_pre_ready, casual_block_size, num_frames_block, base_num_frames_block)
        step_template = torch.cat(
            [
                torch.tensor([999], dtype=torch.int64, device=step_template.device),
                step_template.long(),
                torch.tensor([0], dtype=torch.int64, device=step_template.device),
            ]
        )  # to handle the counter in row works starting from 1
        pre_row = torch.zeros(num_frames_block, dtype=torch.long)
        if num_pre_ready > 0:
            pre_row[: num_pre_ready // casual_block_size] = num_iterations

        while torch.all(pre_row >= (num_iterations - 1)) == False:
            new_row = torch.zeros(num_frames_block, dtype=torch.long)
            for i in range(num_frames_block):
                if i == 0 or pre_row[i - 1] >= (
                    num_iterations - 1
                ):  # the first frame or the last frame is completely denoised
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i - 1] - ar_step
            new_row = new_row.clamp(0, num_iterations)

            update_mask.append(
                (new_row != pre_row) & (new_row != num_iterations)
            )  # False: no need to update， True: need to update
            step_index.append(new_row)
            step_matrix.append(step_template[new_row])
            pre_row = new_row

        # for long video we split into several sequences, base_num_frames is set to the model max length (for training)
        terminal_flag = base_num_frames_block
        if shrink_interval_with_mask:
            idx_sequence = torch.arange(num_frames_block, dtype=torch.int64)
            update_mask = update_mask[0]
            update_mask_idx = idx_sequence[update_mask]
            last_update_idx = update_mask_idx[-1].item()
            terminal_flag = last_update_idx + 1
        # for i in range(0, len(update_mask)):
        for curr_mask in update_mask:
            if terminal_flag < num_frames_block and curr_mask[terminal_flag]:
                terminal_flag += 1
            valid_interval.append((max(terminal_flag - base_num_frames_block, 0), terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)

        if casual_block_size > 1:
            step_update_mask = step_update_mask.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_index = step_index.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_matrix = step_matrix.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            valid_interval = [(s * casual_block_size, e * casual_block_size) for s, e in valid_interval]

        return step_matrix, step_index, step_update_mask, valid_interval

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = "",
        image: PipelineImageInput = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 97,
        num_inference_steps: int = 50,
        shift: float = 1.0,
        guidance_scale: float = 5.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        overlap_history: int = None,
        addnoise_condition: int = 0,
        base_num_frames: int = 97,
        ar_step: int = 5,
        causal_block_size: int = None,
        fps: int = 24,
    ):
        latent_height = height // 8
        latent_width = width // 8
        latent_length = (num_frames - 1) // 4 + 1

        self._guidance_scale = guidance_scale

        i2v_extra_kwrags = {}
        prefix_video = None
        predix_video_latent_length = 0
        if image:
            prefix_video, predix_video_latent_length = self.encode_image(image, height, width, num_frames)

        self.text_encoder.to(self.device)
        prompt_embeds = self.text_encoder.encode(prompt).to(self.transformer.dtype)
        if self.do_classifier_free_guidance:
            negative_prompt_embeds = self.text_encoder.encode(negative_prompt).to(self.transformer.dtype)
        if self.offload:
            self.text_encoder.cpu()

        self.scheduler.set_timesteps(num_inference_steps, device=prompt_embeds.device, shift=shift)
        init_timesteps = self.scheduler.timesteps
        if causal_block_size is None:
            causal_block_size = self.transformer.num_frame_per_block
        fps_embeds = [fps] * prompt_embeds.shape[0]
        fps_embeds = [0 if i == 16 else 1 for i in fps_embeds]
        transformer_dtype = self.transformer.dtype
        # with torch.cuda.amp.autocast(dtype=self.transformer.dtype), torch.no_grad():
        if overlap_history is None or base_num_frames is None or num_frames <= base_num_frames:
            # short video generation
            latent_shape = [16, latent_length, latent_height, latent_width]
            latents = self.prepare_latents(
                latent_shape, dtype=transformer_dtype, device=prompt_embeds.device, generator=generator
            )
            latents = [latents]
            if prefix_video is not None:
                latents[0][:, :predix_video_latent_length] = prefix_video[0].to(transformer_dtype)
            base_num_frames = (base_num_frames - 1) // 4 + 1 if base_num_frames is not None else latent_length
            step_matrix, _, step_update_mask, valid_interval = self.generate_timestep_matrix(
                latent_length, init_timesteps, base_num_frames, ar_step, predix_video_latent_length, causal_block_size
            )
            sample_schedulers = []
            for _ in range(latent_length):
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
                )
                sample_scheduler.set_timesteps(num_inference_steps, device=prompt_embeds.device, shift=shift)
                sample_schedulers.append(sample_scheduler)
            sample_schedulers_counter = [0] * latent_length
            self.transformer.to(self.device)
            for i, timestep_i in enumerate(tqdm(step_matrix)):
                update_mask_i = step_update_mask[i]
                valid_interval_i = valid_interval[i]
                valid_interval_start, valid_interval_end = valid_interval_i
                timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
                latent_model_input = [latents[0][:, valid_interval_start:valid_interval_end, :, :].clone()]
                if addnoise_condition > 0 and valid_interval_start < predix_video_latent_length:
                    noise_factor = 0.001 * addnoise_condition
                    timestep_for_noised_condition = addnoise_condition
                    latent_model_input[0][:, valid_interval_start:predix_video_latent_length] = (
                        latent_model_input[0][:, valid_interval_start:predix_video_latent_length] * (1.0 - noise_factor)
                        + torch.randn_like(latent_model_input[0][:, valid_interval_start:predix_video_latent_length])
                        * noise_factor
                    )
                    timestep[:, valid_interval_start:predix_video_latent_length] = timestep_for_noised_condition
                if not self.do_classifier_free_guidance:
                    noise_pred = self.transformer(
                        torch.stack([latent_model_input[0]]),
                        t=timestep,
                        context=prompt_embeds,
                        fps=fps_embeds,
                        **i2v_extra_kwrags,
                    )[0]
                else:
                    noise_pred_cond = self.transformer(
                        torch.stack([latent_model_input[0]]),
                        t=timestep,
                        context=prompt_embeds,
                        fps=fps_embeds,
                        **i2v_extra_kwrags,
                    )[0]
                    noise_pred_uncond = self.transformer(
                        torch.stack([latent_model_input[0]]),
                        t=timestep,
                        context=negative_prompt_embeds,
                        fps=fps_embeds,
                        **i2v_extra_kwrags,
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                for idx in range(valid_interval_start, valid_interval_end):
                    if update_mask_i[idx].item():
                        latents[0][:, idx] = sample_schedulers[idx].step(
                            noise_pred[:, idx - valid_interval_start],
                            timestep_i[idx],
                            latents[0][:, idx],
                            return_dict=False,
                            generator=generator,
                        )[0]
                        sample_schedulers_counter[idx] += 1
            if self.offload:
                self.transformer.cpu()
                torch.cuda.empty_cache()
            x0 = latents[0].unsqueeze(0)
            videos = self.vae.decode(x0)
            videos = (videos / 2 + 0.5).clamp(0, 1)
            videos = [video for video in videos]
            videos = [video.permute(1, 2, 3, 0) * 255 for video in videos]
            videos = [video.cpu().numpy().astype(np.uint8) for video in videos]
            return videos
        else:
            # long video generation
            base_num_frames = (base_num_frames - 1) // 4 + 1 if base_num_frames is not None else latent_length
            overlap_history_frames = (overlap_history - 1) // 4 + 1
            n_iter = 1 + (latent_length - base_num_frames - 1) // (base_num_frames - overlap_history_frames) + 1
            print(f"n_iter:{n_iter}")
            output_video = None
            for i in range(n_iter):
                if output_video is not None:  # i !=0
                    prefix_video = output_video[:, -overlap_history:].to(prompt_embeds.device)
                    prefix_video = [self.vae.encode(prefix_video.unsqueeze(0))[0]]  # [(c, f, h, w)]
                    if prefix_video[0].shape[1] % causal_block_size != 0:
                        truncate_len = prefix_video[0].shape[1] % causal_block_size
                        print("the length of prefix video is truncated for the casual block size alignment.")
                        prefix_video[0] = prefix_video[0][:, : prefix_video[0].shape[1] - truncate_len]
                    predix_video_latent_length = prefix_video[0].shape[1]
                    finished_frame_num = i * (base_num_frames - overlap_history_frames) + overlap_history_frames
                    left_frame_num = latent_length - finished_frame_num
                    base_num_frames_iter = min(left_frame_num + overlap_history_frames, base_num_frames)
                else:  # i == 0
                    base_num_frames_iter = base_num_frames
                latent_shape = [16, base_num_frames_iter, latent_height, latent_width]
                latents = self.prepare_latents(
                    latent_shape, dtype=transformer_dtype, device=prompt_embeds.device, generator=generator
                )
                latents = [latents]
                if prefix_video is not None:
                    latents[0][:, :predix_video_latent_length] = prefix_video[0].to(transformer_dtype)
                step_matrix, _, step_update_mask, valid_interval = self.generate_timestep_matrix(
                    base_num_frames_iter,
                    init_timesteps,
                    base_num_frames_iter,
                    ar_step,
                    predix_video_latent_length,
                    causal_block_size,
                )
                sample_schedulers = []
                for _ in range(base_num_frames_iter):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
                    )
                    sample_scheduler.set_timesteps(num_inference_steps, device=prompt_embeds.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                sample_schedulers_counter = [0] * base_num_frames_iter
                self.transformer.to(self.device)
                for i, timestep_i in enumerate(tqdm(step_matrix)):
                    update_mask_i = step_update_mask[i]
                    valid_interval_i = valid_interval[i]
                    valid_interval_start, valid_interval_end = valid_interval_i
                    timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
                    latent_model_input = [latents[0][:, valid_interval_start:valid_interval_end, :, :].clone()]
                    if addnoise_condition > 0 and valid_interval_start < predix_video_latent_length:
                        noise_factor = 0.001 * addnoise_condition
                        timestep_for_noised_condition = addnoise_condition
                        latent_model_input[0][:, valid_interval_start:predix_video_latent_length] = (
                            latent_model_input[0][:, valid_interval_start:predix_video_latent_length]
                            * (1.0 - noise_factor)
                            + torch.randn_like(
                                latent_model_input[0][:, valid_interval_start:predix_video_latent_length]
                            )
                            * noise_factor
                        )
                        timestep[:, valid_interval_start:predix_video_latent_length] = timestep_for_noised_condition
                    if not self.do_classifier_free_guidance:
                        noise_pred = self.transformer(
                            torch.stack([latent_model_input[0]]),
                            t=timestep,
                            context=prompt_embeds,
                            fps=fps_embeds,
                            **i2v_extra_kwrags,
                        )[0]
                    else:
                        noise_pred_cond = self.transformer(
                            torch.stack([latent_model_input[0]]),
                            t=timestep,
                            context=prompt_embeds,
                            fps=fps_embeds,
                            **i2v_extra_kwrags,
                        )[0]
                        noise_pred_uncond = self.transformer(
                            torch.stack([latent_model_input[0]]),
                            t=timestep,
                            context=negative_prompt_embeds,
                            fps=fps_embeds,
                            **i2v_extra_kwrags,
                        )[0]
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    for idx in range(valid_interval_start, valid_interval_end):
                        if update_mask_i[idx].item():
                            latents[0][:, idx] = sample_schedulers[idx].step(
                                noise_pred[:, idx - valid_interval_start],
                                timestep_i[idx],
                                latents[0][:, idx],
                                return_dict=False,
                                generator=generator,
                            )[0]
                            sample_schedulers_counter[idx] += 1
                if self.offload:
                    self.transformer.cpu()
                    torch.cuda.empty_cache()
                x0 = latents[0].unsqueeze(0)
                videos = [self.vae.decode(x0)[0]]
                if output_video is None:
                    output_video = videos[0].clamp(-1, 1).cpu()  # c, f, h, w
                else:
                    output_video = torch.cat(
                        [output_video, videos[0][:, overlap_history:].clamp(-1, 1).cpu()], 1
                    )  # c, f, h, w
            output_video = [(output_video / 2 + 0.5).clamp(0, 1)]
            output_video = [video for video in output_video]
            output_video = [video.permute(1, 2, 3, 0) * 255 for video in output_video]
            output_video = [video.cpu().numpy().astype(np.uint8) for video in output_video]
            return output_video
