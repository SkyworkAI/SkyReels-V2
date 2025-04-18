import argparse
import os
import random
import time

import imageio
import torch
from diffusers.utils import load_image

from skyreels_v2_infer.pipelines import Image2VideoPipeline
from skyreels_v2_infer.pipelines import Text2VideoPipeline

MODEL_ID_CONFIG = {
    "text2video": [
        "Skywork/SkyReels-V2-T2V-1.3B-540P",
        "Skywork/SkyReels-V2-T2V-14B-540P",
        "Skywork/SkyReels-V2-T2V-14B-720P",
    ],
    "image2video": [
        "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "Skywork/SkyReels-V2-I2V-14B-540P",
        "Skywork/SkyReels-V2-I2V-14B-720P",
    ],
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="video_out")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-T2V-1.3B-540P")
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--image", type=str, default=None)
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
        default="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface.",
    )
    args = parser.parse_args()

    assert (args.use_usp and args.seed is not None) or (not args.use_usp), "usp mode need seed"
    if args.seed == -1:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))

    image = load_image(args.image).convert("RGB") if args.image else None
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
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

    if image is None:
        assert "T2V" in args.model_id, f"check model_id:{args.model_id}"
        print("init text2video pipeline")
        pipe = Text2VideoPipeline(
            model_path=args.model_id, dit_path=args.model_id, use_usp=args.use_usp, offload=args.offload
        )
    else:
        assert "I2V" in args.model_id, f"check model_id:{args.model_id}"
        print("init img2video pipeline")
        pipe = Image2VideoPipeline(
            model_path=args.model_id, dit_path=args.model_id, use_usp=args.use_usp, offload=args.offload
        )

    kwargs = {
        "prompt": args.prompt,
        "negative_prompt": negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
    }
    if image is None:
        kwargs["height"] = args.height
        kwargs["width"] = args.width
    else:
        kwargs["image"] = load_image(args.image).convert("RGB")
        kwargs["max_area"] = args.height * args.width

    save_dir = os.path.join("result", args.outdir)
    os.makedirs(save_dir, exist_ok=True)

    with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        print(f"infer kwargs:{kwargs}")
        video_frames = pipe(**kwargs)[0]

    if local_rank == 0:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        video_out_file = f"{args.prompt[:100].replace('/','')}_{args.seed}_{current_time}.mp4"
        output_path = os.path.join(save_dir, video_out_file)
        imageio.mimwrite(output_path, video_frames, fps=args.fps, quality=8, output_params=["-loglevel", "error"])
