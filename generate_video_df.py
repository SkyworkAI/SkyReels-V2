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
from diffusers.utils import load_video # 20250425 chaojie prompt travel & video input

from skyreels_v2_infer import DiffusionForcingPipeline
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import PromptEnhancer
from skyreels_v2_infer.pipelines import resizecrop

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="diffusion_forcing")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-DF-1.3B-540P")
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"])
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--video", type=str, default=None) # 20250425 chaojie prompt travel & video input
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

    parser.add_argument("--batch_size", type=int, default=1) # 20250422 pftq: Batch functionality to avoid reloading the model each video
    parser.add_argument("--preserve_image_aspect_ratio", action="store_true")  # 20250422 pftq: Avoid resizing
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards") # 20250422 pftq: expose negative prompt
    
    parser.add_argument(
        "--prompt",
        nargs="+", # 20250425 chaojie prompt travel & video input
        type=str,
        default="A woman in a leather jacket and sunglasses riding a vintage motorcycle through a desert highway at sunset, her hair blowing wildly in the wind as the motorcycle kicks up dust, with the golden sun casting long shadows across the barren landscape.",
    )
    parser.add_argument("--prompt_enhancer", action="store_true")
    parser.add_argument("--teacache", action="store_true")
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="Higher speedup will cause to worse quality -- 0.1 for 2.0x speedup -- 0.2 for 3.0x speedup",
    )
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        help="Using Retention Steps will result in faster generation speed and better generation quality.",
    )
    args = parser.parse_args()

    args.model_id = download_model(args.model_id)
    print("model_id:", args.model_id)

    #20250422 pftq: unneeded with seed synchronization code
    #assert (args.use_usp and args.seed != -1) or (not args.use_usp), "usp mode requires a valid seed"

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
    """
    if args.image:
        args.image = load_image(args.image)
        image_width, image_height = args.image.size
        if image_height > image_width:
            height, width = width, height
        args.image = resizecrop(args.image, height, width)
    image = args.image.convert("RGB") if args.image else None
    """

    # 20250425 chaojie prompt travel & video input
    video = []
    if args.video:
        pre_video_length = 17
        if args.overlap_history is not None:
            pre_video_length = args.overlap_history
        args.video = load_video(args.video) 
        arg_width = width
        arg_height = height
        for img in args.video:
            # 20250422 pftq: option to preserve image aspect ratio
            if args.preserve_image_aspect_ratio:
                img_width, img_height = img.size
                if img_height > img_width:
                    height, width = width, height
                    width = int(height / img_height * img_width)
                else:
                    height = int(width / img_width * img_height)
        
                divisibility=16
                if width%divisibility!=0:
                        width = width - (width%divisibility)
                if height%divisibility!=0:
                        height = height - (height%divisibility)
        
                img = resizecrop(img, height, width)
            else:
                image_width, image_height = img.size
                if image_height > image_width:
                    height, width = width, height
                img = resizecrop(img, height, width)
            video.append(img.convert("RGB").resize((width, height)))
            video = video[-pre_video_length:]
    else:
        video = None

        #20250422 pftq: Add error handling for image loading, aspect ratio preservation
    image = None
    if args.image and not args.video:
        try:
            image = load_image(args.image).convert("RGB")

            # 20250422 pftq: option to preserve image aspect ratio
            if args.preserve_image_aspect_ratio:
                img_width, img_height = image.size
                if img_height > img_width:
                    height, width = width, height
                    width = int(height / img_height * img_width)
                else:
                    height = int(width / img_width * img_height)

                divisibility=16
                if width%divisibility!=0:
                        width = width - (width%divisibility)
                if height%divisibility!=0:
                        height = height - (height%divisibility)

                image = resizecrop(image, height, width)
            else:
                image_width, image_height = image.size
                if image_height > image_width:
                    height, width = width, height
                image = resizecrop(image, height, width)
        except Exception as e:
            raise ValueError(f"Failed to load or process image: {e}")
    
    #negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    negative_prompt = args.negative_prompt # 20250422 pftq: allow editable negative prompt
  
    save_dir = os.path.join("result", args.outdir)
    os.makedirs(save_dir, exist_ok=True)
    local_rank = 0
    if args.use_usp:
        assert (
            not args.prompt_enhancer
        ), "`--prompt_enhancer` is not allowed if using `--use_usp`. We recommend running the skyreels_v2_infer/pipelines/prompt_enhancer.py script first to generate enhanced prompt before enabling the `--use_usp` parameter."
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

    prompt_input = args.prompt
    if args.prompt_enhancer and args.image is None:
        print(f"init prompt enhancer")
        prompt_enhancer = PromptEnhancer()
        prompt_input = prompt_enhancer(prompt_input)
        print(f"enhanced prompt: {prompt_input}")
        del prompt_enhancer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Rank {local_rank}: {width}x{height} | Image Input: "+str(image!=None) + " | Video Input: "+str(video!=None))
    
    # 20250423 pftq: fixed 20-min load times on multi-GPU caused by contention, reduced down to 12 min roughly the same as single GPU.
    print("Initializing pipe at "+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    starttime = time.time()
    pipe = DiffusionForcingPipeline(
        args.model_id,
        dit_path=args.model_id,
        device=torch.device("cuda"),
        weight_dtype=torch.bfloat16,
        use_usp=args.use_usp,
        offload=args.offload,
    )
    totaltime = time.time()-starttime
    print("Finished initializing pipe at "+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())+" ("+str(int(totaltime))+" seconds)")

    if args.causal_attention:
        pipe.transformer.set_ar_attention(args.causal_block_size)
    
    if args.teacache:
        if args.ar_step > 0:
            num_steps = args.inference_steps + (((args.base_num_frames - 1)//4 + 1) // args.causal_block_size - 1) * args.ar_step
            print('num_steps:', num_steps)
        else:
            num_steps = args.inference_steps
        pipe.transformer.initialize_teacache(enable_teacache=True, num_steps=num_steps, 
                                             teacache_thresh=args.teacache_thresh, use_ret_steps=args.use_ret_steps, 
                                             ckpt_dir=args.model_id)

    #20250422 pftq: Set preferred linear algebra backend to avoid cuSOLVER issues
    torch.backends.cuda.preferred_linalg_library("default")  # or try "magma" if available
    
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


        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        #video_out_file = f"{args.prompt[:100].replace('/','')}_{args.seed}_{current_time}.mp4"
        
        # 20250422 pftq: more useful filename
        gpucount = ""
        if args.use_usp and dist.get_world_size():
            gpucount = "_"+str(dist.get_world_size())+"xGPU"
        prompt_summary = ""
        if type(args.prompt) is list:
            prompt_summary = args.prompt[0][:10].replace('/','')
        else:
            prompt_summary = args.prompt[:10].replace('/','')
        video_out_file = f"{current_time}_skyreels2df_{args.resolution}-{args.num_frames}f_cfg{args.guidance_scale}_steps{args.inference_steps}_seed{args.seed}{gpucount}_{prompt_summary}_{idx}.mp4" 
        
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

                # 20250425 chaojie prompt travel & video input
                video=video,
                local_rank=local_rank,
                save_dir=save_dir,
                video_out_file=video_out_file,
            )[0]
    
        if local_rank == 0:
            output_path = os.path.join(save_dir, video_out_file)
            imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
