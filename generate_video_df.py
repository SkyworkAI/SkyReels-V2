import argparse
import gc
import os
import random
import time
import torch
from diffusers.utils import load_image
import imageio
from PIL import Image  #20250422 pftq: Added for image resizing and cropping
import numpy as np  #20250422 pftq: Added for seed synchronization

from skyreels_v2_infer import DiffusionForcingPipeline
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import PromptEnhancer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="diffusion_forcing")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-DF-1.3B-540P")
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"])
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--ar_step", type=int, default=0)
    parser.add_argument("--causal_attention", action="store_true")
    parser.add_argument("--causal_block_size", type=int, default=1)
    parser.add_argument("--base_num_frames", type=int, default=97)
    parser.add_argument("--overlap_history", type=int, default=None)
    parser.add_argument("--addnoise_condition", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--use_usp", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A woman in a leather jacket and sunglasses riding a vintage motorcycle through a desert highway at sunset, her hair blowing wildly in the wind as the motorcycle kicks up dust, with the golden sun casting long shadows across the barren landscape.",
    )
    parser.add_argument("--prompt_enhancer", action="store_true")
    
    parser.add_argument("--batch_size", type=int, default=1) # 20250422 pftq: Batch functionality to avoid reloading the model each video
    parser.add_argument("--preserve_image_aspect_ratio", action="store_true")  # 20250422 pftq: Avoid resizing
    
    args = parser.parse_args()

    args.model_id = download_model(args.model_id)
    print("model_id:", args.model_id)

    #20250422 pftq: unneeded with seed synchronization code
    #assert (args.use_usp and args.seed != -1) or (not args.use_usp), "usp mode requires a valid seed"

    local_rank = 0
    if args.use_usp:
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        import torch.distributed as dist

        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(dist.get_rank())
        device = "cuda"

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )

    if args.resolution == "540P":
        height = 544
        width = 960
    elif args.resolution == "720P":
        height = 720
        width = 1280
    else:
        raise ValueError(f"Invalid resolution: {args.resolution}")

    num_frames = args.num_frames
    fps = args.fps

    if num_frames > args.base_num_frames:
        assert (
            args.overlap_history is not None
        ), 'You are supposed to specify the "overlap_history" to support the long video generation. 17 and 37 are recommended to set.'
    if args.addnoise_condition > 60:
        print(
            f'You have set "addnoise_condition" as {args.addnoise_condition}. The value is too large which can cause inconsistency in long video generation. The value is recommended to set 20.'
        )

    guidance_scale = args.guidance_scale
    shift = args.shift
    #image = load_image(args.image).convert("RGB") if args.image else None

    #20250422 pftq: Add error handling for image loading, aspect ratio preservation, and multi-GPU synchronization
    image = None
    if args.image:
        if local_rank == 0:
            try:
                image = load_image(args.image).convert("RGB")

                # 20250422 pftq: option to preserve image aspect ratio
                if args.preserve_image_aspect_ratio:
                    img_width, img_height = image.size
                    height = int(width / img_width * img_height)
            except Exception as e:
                raise ValueError(f"Failed to load or process image: {e}")
                
        if args.use_usp:
            # Broadcast image to other ranks
            image_data = torch.tensor(np.array(image), dtype=torch.uint8, device="cuda") if image is not None else None
            if local_rank == 0:
                dist.broadcast(image_data, src=0)
            else:
                image_data = torch.empty((height, width, 3), dtype=torch.uint8, device="cuda")
                dist.broadcast(image_data, src=0)
                image = Image.fromarray(image_data.cpu().numpy())

            #20250422 pftq: Broadcast height and width to ensure consistency
            height_tensor = torch.tensor(height, dtype=torch.int64, device="cuda")
            width_tensor = torch.tensor(width, dtype=torch.int64, device="cuda")
            dist.broadcast(height_tensor, src=0)
            dist.broadcast(width_tensor, src=0)
            height = height_tensor.item()
            width = width_tensor.item()

    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    save_dir = os.path.join("result", args.outdir)
    os.makedirs(save_dir, exist_ok=True)

    prompt_input = args.prompt
    if args.prompt_enhancer and args.image is None:
        print(f"init prompt enhancer")
        prompt_enhancer = PromptEnhancer()
        prompt_input = prompt_enhancer(prompt_input)
        print(f"enhanced prompt: {prompt_input}")
        del prompt_enhancer
        gc.collect()
        torch.cuda.empty_cache()

    pipe = DiffusionForcingPipeline(
        args.model_id,
        dit_path=args.model_id,
        device=torch.device("cuda"),
        weight_dtype=torch.bfloat16,
        use_usp=args.use_usp,
        offload=args.offload,
    )

    if args.causal_attention:
        pipe.transformer.set_ar_attention(args.causal_block_size)

    for idx in range(args.batch_size): # 20250422 pftq: implemented --batch_size
        if local_rank == 0:
            print(f"prompt:{prompt_input}")
            print(f"guidance_scale:{guidance_scale}")
            print(f"Generating video {idx+1} of {args.batch_size}")

        #20250422 pftq: Synchronize seed across all ranks
        if args.use_usp:
            try:
                #20250422 pftq: Synchronize ranks before seed broadcasting
                dist.barrier()

                #20250422 pftq: Always broadcast seed to ensure consistency
                if local_rank == 0:
                    if args.seed == -1 or idx > 0:
                        args.seed = int(random.randrange(4294967294))
                seed_tensor = torch.tensor(args.seed, dtype=torch.int64, device="cuda")
                dist.broadcast(seed_tensor, src=0)
                args.seed = seed_tensor.item()

                #20250422 pftq: Synchronize ranks after seed broadcasting
                dist.barrier()
            except Exception as e:
                print(f"[Rank {local_rank}] Seed broadcasting error: {e}")
                dist.destroy_process_group()
                raise

        else:
            #20250422 pftq: Single GPU seed initialization
            if args.seed == -1 or idx > 0:
                args.seed = int(random.randrange(4294967294))

        #20250422 pftq: Set seeds for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
            video_frames = pipe(
                prompt=prompt_input,
                negative_prompt=negative_prompt,
                image=image,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=args.inference_steps,
                shift=shift,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                overlap_history=args.overlap_history,
                addnoise_condition=args.addnoise_condition,
                base_num_frames=args.base_num_frames,
                ar_step=args.ar_step,
                causal_block_size=args.causal_block_size,
                fps=fps,
            )[0]
    
        if local_rank == 0:
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            #video_out_file = f"{args.prompt[:100].replace('/','')}_{args.seed}_{current_time}.mp4"
            
            # 20250422 pftq: more useful filename
            video_out_file = f"{current_time}_cfg{guidance_scale}_steps{args.inference_steps}_seed{args.seed}_{args.prompt[:100].replace('/','')}_{idx}.mp4"
            
            output_path = os.path.join(save_dir, video_out_file)
            imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
