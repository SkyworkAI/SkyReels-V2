<p align="center">
  <img src="assets/logo2.png" alt="SkyReels Logo" width="50%">
</p>

<h1 align="center">SkyReels V2: Infinite-Length Film Generative Model</h1> 

<p align="center">
📑 <a href="https://arxiv.org/pdf/2504.13074">Technical Report</a> · 👋 <a href="https://www.skyreels.ai/home?utm_campaign=github_V2" target="_blank">Playground</a> · 💬 <a href="https://discord.gg/PwM6NYtccQ" target="_blank">Discord</a> · 🤗 <a href="https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9" target="_blank">Hugging Face</a> · 🤖 <a href="https://www.modelscope.cn/collections/SkyReels-V2-f665650130b144" target="_blank">ModelScope</a>
</p>

---
Welcome to the SkyReels V2 repository! Here, you'll find the model weights and inference code for our infinite-lenght film genetative models


## 🔥🔥🔥 News!!

* Apr 21, 2025: 👋 We release the inference code and model weights of [SkyReels-V2](https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9) Series Models.
* Apr 3, 2025: 🔥 We also release [SkyReels-A2](https://github.com/SkyworkAI/SkyReels-A2). This is an open-sourced controllable video generation framework capable of assembling arbitrary visual elements.

## 🎥 Demos
<table>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/f6f9f9a7-5d5f-433c-9d73-d8d593b7ad25" width="100%"></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/0eb13415-f4d9-4aaf-bcd3-3031851109b9" width="100%"></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/dcd16603-5bf4-4786-8e4d-1ed23889d07a" width="100%"></video>
    </td>
  </tr>
</table>
The demos above showcase 30-second videos generated using our SkyReels-V2 Diffusion Forcing model.


## 📑 TODO List

- [x] Technique Report
- [x] Checkpoints of the 14B and 1.3B Models Series
- [x] Single-GPU & Multi-GPU Inference Code
- [x] SkyCaptioner-V1 Video Annotation Model
- [x] Prompt Rewrite
- [ ] Diffusers integration
- [ ] Checkpoints of the 5B Models Series
- [ ] Checkpoints of the Camera Director Models
- [ ] Checkpoints of the Step & Guidance Distill Model


## 🚀 Quickstart

#### Installation
```shell
# clone the repository.
git clone https://github.com/SkyworkAI/Skyreels-V2
cd Skyreels-V2
# Install dependencies. Test environment uses Python 3.10.12.
pip install -r requirements.txt
```

#### Model Download
You can download our models from Hugging Face:
<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Model Variant</th>
      <th>Recommended Height/Width/Frame</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5">Diffusion Forcing</td>
      <td>1.3B-540P</td>
      <td>544 * 960 * 97f</td>
      <td>🤗 <a href="https://huggingface.co/SkyworkAI/SkyReels-V2-DF-1.3B">Huggingface</a> 🤖 <a href="https://www.modelscope.cn/SkyworkAI/SkyReels-V2-DF-1.3B">ModelScope</a></td>
    </tr>
    <tr>
      <td>5B-540P</td>
      <td>544 * 960 * 97f</td>
      <td>Coming Soon</td>
    </tr>
    <tr>
      <td>5B-720P</td>
      <td>720 * 1280 * 121f</td>
      <td>Coming Soon</td>
    </tr>
    <tr>
      <td>14B-540P</td>
      <td>544 * 960 * 97f</td>
      <td>🤗 <a href="https://huggingface.co/SkyworkAI/SkyReels-V2-DF-14B-540P">Huggingface</a> 🤖 <a href="https://www.modelscope.cn/SkyworkAI/SkyReels-V2-DF-14B-540P">ModelScope</a></td>
    </tr>
    <tr>
      <td>14B-720P</td>
      <td>720 * 1280 * 121f</td>
      <td>🤗 <a href="https://huggingface.co/SkyworkAI/SkyReels-V2-DF-14B-720P">Huggingface</a> 🤖 <a href="https://www.modelscope.cn/SkyworkAI/SkyReels-V2-DF-14B-720P">ModelScope</a></td>
    </tr>
    <tr>
      <td rowspan="5">Text-to-Video</td>
      <td>1.3B-540P</td>
      <td>544 * 960 * 97f</td>
      <td>🤗 <a href="https://huggingface.co/SkyworkAI/SkyReels-V2-T2V-1.3B">Huggingface</a> 🤖 <a href="https://www.modelscope.cn/SkyworkAI/SkyReels-V2-T2V-1.3B">ModelScope</a></td>
    </tr>
    <tr>
      <td>5B-540P</td>
      <td>544 * 960 * 97f</td>
      <td>Coming Soon</td>
    </tr>
    <tr>
      <td>5B-720P</td>
      <td>720 * 1280 * 121f</td>
      <td>Coming Soon</td>
    </tr>
    <tr>
      <td>14B-540P</td>
      <td>544 * 960 * 97f</td>
      <td>🤗 <a href="https://huggingface.co/SkyworkAI/SkyReels-V2-T2V-14B-540P">Huggingface</a> 🤖 <a href="https://www.modelscope.cn/SkyworkAI/SkyReels-V2-T2V-14B-540P">ModelScope</a></td>
    </tr>
    <tr>
      <td>14B-720P</td>
      <td>720 * 1280 * 121f</td>
      <td>🤗 <a href="https://huggingface.co/SkyworkAI/SkyReels-V2-T2V-14B-720P">Huggingface</a> 🤖 <a href="https://www.modelscope.cn/SkyworkAI/SkyReels-V2-T2V-14B-720P">ModelScope</a></td>
    </tr>
    <tr>
      <td rowspan="5">Image-to-Video</td>
      <td>1.3B-540P</td>
      <td>544 * 960 * 97f</td>
      <td>🤗 <a href="https://huggingface.co/SkyworkAI/SkyReels-V2-I2V-1.3B">Huggingface</a> 🤖 <a href="https://www.modelscope.cn/SkyworkAI/SkyReels-V2-I2V-1.3B">ModelScope</a></td>
    </tr>
    <tr>
      <td>5B-540P</td>
      <td>544 * 960 * 97f</td>
      <td>Coming Soon</td>
    </tr>
    <tr>
      <td>5B-720P</td>
      <td>720 * 1280 * 121f</td>
      <td>Coming Soon</td>
    </tr>
    <tr>
      <td>14B-540P</td>
      <td>544 * 960 * 97f</td>
      <td>🤗 <a href="https://huggingface.co/SkyworkAI/SkyReels-V2-I2V-14B-540P">Huggingface</a> 🤖 <a href="https://www.modelscope.cn/SkyworkAI/SkyReels-V2-I2V-14B-540P">ModelScope</a></td>
    </tr>
    <tr>
      <td>14B-720P</td>
      <td>720 * 1280 * 121f</td>
      <td>🤗 <a href="https://huggingface.co/SkyworkAI/SkyReels-V2-I2V-14B-720P">Huggingface</a> 🤖 <a href="https://www.modelscope.cn/SkyworkAI/SkyReels-V2-I2V-14B-720P">ModelScope</a></td>
    </tr>
    <tr>
      <td rowspan="3">Camera Director</td>
      <td>5B-540P</td>
      <td>544 * 960 * 97f</td>
      <td>Coming Soon</td>
    </tr>
    <tr>
      <td>5B-720P</td>
      <td>720 * 1280 * 121f</td>
      <td>Coming Soon</td>
    </tr>
    <tr>
      <td>14B-720P</td>
      <td>720 * 1280 * 121f</td>
      <td>Coming Soon</td>
    </tr>
  </tbody>
</table>

After downloading, set the model path in your generation commands:


#### Single GPU Inference

- **Diffusion Forcing**

The <a href="https://arxiv.org/abs/2407.01392">**Diffusion Forcing**</a> version model allows us to generate Infinite-Length videos. This model supports both text-to-video (t2v) and image-to-video (i2v) tasks, and it can perform inference in both synchronous and asynchronous modes.

```shell
model_id=Skywork/SkyReels-V2-DF-14B-540P
# synchronous inference
python3 generate_video_df.py \
  --model_id ${model_id} \
  --height 544 \
  --width 960 \
  --ar_step 0 \
  --base_num_frames 97 \
  --num_frames 257 \
  --overlap_history 17 \
  --prompt "A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface." \
  --offload
```
> **Note**: 
> - If you want to run the DF model for **image-to-video (I2V)** tasks, add `--image ${image_path}` to your command.
> - You can use `--ar_step 5` to enable asynchronous inference. When asynchronous inference, `--causal_block_size 5` is recommanded.
> - `--addnoise_condition 20` is recommanded to add for better long video consistency.
> - To reduce peak VRAM, lower the `--base_num_frames` for the same generative length `--num_frames`. This may slightly reduce video quality.
> - You can use `--prompt_enhance` to expand the prompt into a more detailed description, which is only supported for text-to-video.

- **Text To Video & Image To Video**

```shell
# run Text-to-Video Generation
model_id=Skywork/SkyReels-V2-T2V-14B-540P
python3 generate_video.py \
  --model_id ${model_id} \
  --height 544 \
  --width 960 \
  --num_frames 97 \
  --guidance_scale 6.0 \
  --shift 8.0 \
  --fps 24 \
  --prompt "A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface." \
  --offload
```
> **Note**: 
> - When using an **image-to-video (I2V)** model, you must provide an input image using the `--image  ${image_path}` parameter.
> - You can use `--prompt_enhance` to expand the prompt into a more detailed description, which is only supported for text-to-video.

- **Prompt Rewriter**

The prompt rewriter is implemented based on <a href="https://huggingface.co/Qwen/Qwen2.5-32B-Instruct">Qwen2.5-32B-Instruct</a> and  is utilized via the `--prompt_enhance` parameter. It works ideally for short prompts, while for long prompts, it might generate an excessively lengthy prompt that could lead to over-saturation in the generative video.

**Advanced Configuration Options**

Below are the key parameters you can customize for video generation:

| Parameter | Recommended Value | Description |
|:----------------------:|:---------:|:-----------------------------------------:|
| --prompt |  | Text description for generating your video |
| --image |  | Path to input image for image-to-video generation |
| --resolution | 540P or 720P | Output video resolution (select based on model type) |
| --num_frames | 97 or 121 | Total frames to generate (**97 for 540P models**, **121 for 720P models**) |
| --inference_steps | 50 | Number of denoising steps |
| --fps | 24 | Frames per second in the output video |
| --shift | 8.0 or 5.0 | Flow matching scheduler parameter (**8.0 for T2V**, **5.0 for I2V**) |
| --guidance_scale | 6.0 or 5.0 | Controls text adherence strength (**6.0 for T2V**, **5.0 for I2V**) |
| --seed |  | Fixed seed for reproducible results (omit for random generation) |
| --offload | True | Offloads model components to CPU to reduce VRAM usage (recommended) |
| --use_usp | True | Enables multi-GPU acceleration with xDiT USP |
| --outdir | ./video_out | Directory where generated videos will be saved |

**Diffusion Forcing Additional Parameters**
| Parameter | Recommended Value | Description |
|:----------------------:|:---------:|:-----------------------------------------:|
| --ar_step | 0 | Controls asynchronous inference (0 for synchronous mode) |
| --base_num_frames | 97 or 121 | Base frame count (**97 for 540P**, **121 for 720P**) |
| --overlap_history | 17 | Number of frames to overlap for smooth transitions in long videos |
| --addnoise_condition | 20 | Improves consistency in long video generation |
| --causal_block_size | 5 | Recommended when using asynchronous inference (--ar_step > 0) |

#### Multi-GPU inference using xDiT USP

We use [xDiT](https://github.com/xdit-project/xDiT) USP to accelerate inference.  For example, to generate a video with 2 GPUs, you can use the following command:
- **Diffusion Forcing**
```shell
model_id=Skywork/SkyReels-V2-DF-14B-540P
# diffusion forcing synchronous inference
torchrun --nproc_per_node=2 generate_video_df.py \
  --model_id ${model_id} \
  --height 544 \
  --width 960 \
  --ar_step 0 \
  --base_num_frames 97 \
  --num_frames 257 \
  --overlap_history 17 \
  --prompt "A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface." \
  --use_usp \
  --offload \
  --seed 42
```
- **Text To Video & Image To Video**
```shell
# run Text-to-Video Generation
model_id=Skywork/SkyReels-V2-T2V-14B-540P
torchrun --nproc_per_node=2 generate_video.py \
  --model_id ${model_id} \
  --height 544 \
  --width 960 \
  --num_frames 97 \
  --guidance_scale 6.0 \
  --shift 8.0 \
  --fps 24 \
  --offload \
  --prompt "A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface." \
  --use_usp \
  --seed 42
```

## Contents
  - [Abstract](#abstract)
  - [Methodology of SkyReels-V2](#methodology-of-skyreels-v2)
  - [Key Contributions of SkyReels-V2](#key-contributions-of-skyreels-v2)
    - [Skyreels Captioner](#skyreels-captioner)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Diffusion Forcing](#diffusion-forcing)
    - [Hight-Quality Supervised Fine-Tuning(SFT)](#high-quality-supervised-fine-tuning-sft)
  - [Performance](#performance)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)
---

## Abstract
Recent advances in video generation have been driven by diffusion models and autoregressive frameworks, yet critical challenges persist in harmonizing prompt adherence, visual quality, motion dynamics, and duration: compromises in motion dynamics to enhance temporal visual quality, constrained video duration (5-10 seconds) to prioritize resolution, and inadequate shot-aware generation stemming from general-purpose MLLMs' inability to interpret cinematic grammar, such as shot composition, actor expressions, and camera motions. These intertwined limitations hinder realistic long-form synthesis and professional film-style generation. 

To address these limitations, we introduce SkyReels-V2, the world's first infinite-length film generative model using a Diffusion Forcing framework. Our approach synergizes Multi-modal Large Language Models (MLLM), Multi-stage Pretraining, Reinforcement Learning, and Diffusion Forcing techniques to achieve comprehensive optimization. Beyond its technical innovations, SkyReels-V2 enables multiple practical applications, including Story Generation, Image-to-Video Synthesis, Camera Director functionality, and multi-subject consistent video generation through our <a href="https://github.com/SkyworkAI/SkyReels-A2">Skyreels-A2</a> system.

## Methodology of SkyReels-V2

The SkyReels-V2 methodology consists of several interconnected components. It starts with a comprehensive data processing pipeline that prepares various quality training data. At its core is the Video Captioner architecture, which provides detailed annotations for video content. The system employs a multi-task pretraining strategy to build fundamental video generation capabilities. Post-training optimization includes Reinforcement Learning to enhance motion quality, Diffusion Forcing Training for generating extended videos, and High-quality Supervised Fine-Tuning (SFT) stages for visual refinement. The model runs on optimized computational infrastructure for efficient training and inference. SkyReels-V2 supports multiple applications, including Story Generation, Image-to-Video Synthesis, Camera Director functionality, and Elements-to-Video Generation.

<p align="center">
  <img src="assets/main_pipeline.jpg" alt="mainpipeline" width="100%">
</p>

## Key Contributions of SkyReels-V2

#### Skyreels Captioner

SkyCaptioner-V1 serves as our video captioning model for data annotation. This model is trained on the captioning result from the base model <a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">Qwen2.5-VL-72B-Instruct</a> and the sub-expert captioners on a balanced video data. The balanced video data is a carefully curated dataset of approximately 2 million videos—selected from an initial pool of 10 million samples to ensure conceptual balance and annotation quality. Built upon the <a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">Qwen2.5-VL-7B-Instruct</a> foundation model, SkyCaptioner-V1 is fine-tuned to enhance performance in domain-specific video captioning tasks. To compare the performance with the SOTA models, we conducted a manual assessment of accuracy across different captioning fields using a test set of 1,000 samples. The proposed SkyCaptioner-V1 achieves the highest average accuracy among the baseline models, and show a dramatic result in the shot related fields

<p align="center">
<table align="center">
  <thead>
    <tr>
      <th>model</th>
      <th><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">Qwen2.5-VL-7B-Ins.</a></th>
      <th><a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">Qwen2.5-VL-72B-Ins.</a></th>
      <th><a href="https://huggingface.co/omni-research/Tarsier2-Recap-7b">Tarsier2-Recap-7b</a></th>
      <th>SkyCaptioner-V1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Avg accuracy</td>
      <td>51.4%</td>
      <td>58.7%</td>
      <td>49.4%</td>
      <td><strong>76.3%</strong></td>
    </tr>
    <tr>
      <td>shot type</td>
      <td>76.8%</td>
      <td>82.5%</td>
      <td>60.2%</td>
      <td><strong>93.7%</strong></td>
    </tr>
    <tr>
      <td>shot angle</td>
      <td>60.0%</td>
      <td>73.7%</td>
      <td>52.4%</td>
      <td><strong>89.8%</strong></td>
    </tr>
    <tr>
      <td>shot position</td>
      <td>28.4%</td>
      <td>32.7%</td>
      <td>23.6%</td>
      <td><strong>83.1%</strong></td>
    </tr>
    <tr>
      <td>camera motion</td>
      <td>62.0%</td>
      <td>61.2%</td>
      <td>45.3%</td>
      <td><strong>85.3%</strong></td>
    </tr>
    <tr>
      <td>expression</td>
      <td>43.6%</td>
      <td>51.5%</td>
      <td>54.3%</td>
      <td><strong>68.8%</strong></td>
    </tr>
    <tr>
      <td colspan="5" style="text-align: center; border-bottom: 1px solid #ddd; padding: 8px;"></td>
    </tr>
    <tr>
      <td>TYPES_type</td>
      <td>43.5%</td>
      <td>49.7%</td>
      <td>47.6%</td>
      <td><strong>82.5%</strong></td>
    </tr>
    <tr>
      <td>TYPES_sub_type</td>
      <td>38.9%</td>
      <td>44.9%</td>
      <td>45.9%</td>
      <td><strong>75.4%</strong></td>
    </tr>
    <tr>
      <td>appearance</td>
      <td>40.9%</td>
      <td>52.0%</td>
      <td>45.6%</td>
      <td><strong>59.3%</strong></td>
    </tr>
    <tr>
      <td>action</td>
      <td>32.4%</td>
      <td>52.0%</td>
      <td><strong>69.8%</strong></td>
      <td>68.8%</td>
    </tr>
    <tr>
      <td>position</td>
      <td>35.4%</td>
      <td>48.6%</td>
      <td>45.5%</td>
      <td><strong>57.5%</strong></td>
    </tr>
    <tr>
      <td>is_main_subject</td>
      <td>58.5%</td>
      <td>68.7%</td>
      <td>69.7%</td>
      <td><strong>80.9%</strong></td>
    </tr>
    <tr>
      <td>environment</td>
      <td>70.4%</td>
      <td><strong>72.7%</strong></td>
      <td>61.4%</td>
      <td>70.5%</td>
    </tr>
    <tr>
      <td>lighting</td>
      <td>77.1%</td>
      <td><strong>80.0%</strong></td>
      <td>21.2%</td>
      <td>76.5%</td>
    </tr>
  </tbody>
</table>
</p>

#### Reinforcement Learning
Inspired by the previous success in LLM, we propose to enhance the performance of the generative model by Reinforcement Learning. Specifically, we focus on the motion quality because we find that the main drawback of our generative model is:

- the generative model does not handle well with large, deformable motions. 
- the generated videos may violate the physical law.

To avoid the degradation in other metrics, such as text alignment and video quality, we ensure the preference data pairs have comparable text alignment and video quality, while only the motion quality varies. This requirement poses greater challenges in obtaining preference annotations due to the inherently higher costs of human annotation. To addressthis challenge, we propose a semi-automatic pipeline that strategically combines automatically generated motion pairsand human annotation results. This hybrid approach not only enhances the data scale but also improves alignmentwith human preferences through curated quality control. Leveraging this enhanced dataset, we first train a specializedreward model to capture the generic motion quality differences between paired samples. This learned reward function subsequently guides the sample selection process for Direct Preference Optimization (DPO), enhancing the motionquality of the generative model.

#### Diffusion Forcing

We introduce the Diffusion Forcing Transformer to unlock our model’s ability to generate long videos. Diffusion Forcing is a training and sampling strategy where each token is assigned an independent noise level. This allows tokens to be denoised according to arbitrary, per-token schedules. Conceptually, this approach functions as a form of partial masking: a token with zero noise is fully unmasked, while complete noise fully masks it. Diffusion Forcing trains the model to "unmask" any combination of variably noised tokens, using the cleaner tokens as conditional information to guide the recovery of noisy ones. Building on this, our Diffusion Forcing Transformer can extend video generation indefinitely based on the last frames of the previous segment. Note that the synchronous full sequence diffusion is a special case of Diffusion Forcing, where all tokens share the same noise level. This relationship allows us to fine-tune the Diffusion Forcing Transformer from a full-sequence diffusion model.

#### High-Quality Supervised Fine-Tuning (SFT)

We implement two sequential high-quality supervised fine-tuning (SFT) stages at 540p and 720p resolutions respectively,with the initial SFT phase conducted immediately after pretraining but prior to reinforcement learning (RL) stage.This first-stage SFT serves as a conceptual equilibrium trainer, building upon the foundation model’s pretraining outcomes that utilized only fps24 video data, while strategically removing FPS embedding components to streamline thearchitecture. Trained with the high-quality concept-balanced samples, this phase establishes optimized initialization parameters for subsequent training processes. Following this, we execute a secondary high-resolution SFT at 720p after completing the diffusion forcing stage, incorporating identical loss formulations and thehigher-quality concept-balanced datasets by the manually filter. This final refinement phase focuses on resolution increase such that the overall video quality will be further enhanced.

## Performance

To comprehensively evaluate our proposed method, we construct the SkyReels-Bench for human assessment and leveraged the open-source <a href="https://github.com/Vchitect/VBench">V-Bench</a> for automated evaluation. This allows us to compare our full-sequence diffusion model, diffusion-forcing variant, and state-of-the-art (SOTA) baselines, including both open-source and proprietary models.

#### Human Evaluation

For human evaluation, we design SkyReels-Bench with 1,020 text prompts, systematically assessing three dimensions: Instruction Adherence, Motion Quality, Consistency and Visual Quality. This benchmark is designed to evaluate both text-to-video (T2V) and image-to-video (I2V) generation models, providing comprehensive assessment across different generation paradigms. To ensure fairness, all models were evaluated under default settings with consistent resolutions, and no post-generation filtering was applied.

- Text To Video Models

<p align="center">
<table align="center">
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Average</th>
      <th>Instruction Adherence</th>
      <th>Consistency</th>
      <th>Visual Quality</th>
      <th>Motion Quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://runwayml.com/research/introducing-gen-3-alpha">Runway-Gen3 Alpha</a></td>
      <td>2.53</td>
      <td>2.19</td>
      <td>2.57</td>
      <td>3.23</td>
      <td>2.11</td>
    </tr>
    <tr>
      <td><a href="https://github.com/Tencent/HunyuanVideo">HunyuanVideo-13B</a></td>
      <td>2.82</td>
      <td>2.64</td>
      <td>2.81</td>
      <td>3.20</td>
      <td>2.61</td>
    </tr>
    <tr>
      <td><a href="https://klingai.com">Kling-1.6 STD Mode</a></td>
      <td>2.99</td>
      <td>2.77</td>
      <td>3.05</td>
      <td>3.39</td>
      <td><strong>2.76</strong></td>
    </tr>
    <tr>
      <td><a href="https://hailuoai.video">Hailuo-01</a></td>
      <td>3.0</td>
      <td>2.8</td>
      <td>3.08</td>
      <td>3.29</td>
      <td>2.74</td>
    </tr>
    <tr>
      <td><a href="https://github.com/Wan-Video/Wan2.1">Wan2.1-14B</a></td>
      <td>3.12</td>
      <td>2.91</td>
      <td>3.31</td>
      <td><strong>3.54</strong></td>
      <td>2.71</td>
    </tr>
    <tr>
      <td>SkyReels-V2</td>
      <td><strong>3.14</strong></td>
      <td><strong>3.15</strong></td>
      <td><strong>3.35</strong></td>
      <td>3.34</td>
      <td>2.74</td>
    </tr>
  </tbody>
</table>
</p>

The evaluation demonstrates that our model achieves significant advancements in **instruction adherence (3.15)** compared to baseline methods, while maintaining competitive performance in **motion quality (2.74)** without sacrificing the **consistency (3.35)**. 

- Image To Video Models

<p align="center">
<table align="center">
  <thead>
    <tr>
      <th>Model</th>
      <th>Average</th>
      <th>Instruction Adherence</th>
      <th>Consistency</th>
      <th>Visual Quality</th>
      <th>Motion Quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/Tencent/HunyuanVideo">HunyuanVideo-13B</a></td>
      <td>2.84</td>
      <td>2.97</td>
      <td>2.95</td>
      <td>2.87</td>
      <td>2.56</td>
    </tr>
    <tr>
      <td><a href="https://github.com/Wan-Video/Wan2.1">Wan2.1-14B</a></td>
      <td>2.85</td>
      <td>3.10</td>
      <td>2.81</td>
      <td>3.00</td>
      <td>2.48</td>
    </tr>
    <tr>
      <td><a href="https://hailuoai.video">Hailuo-01</a></td>
      <td>3.05</td>
      <td>3.31</td>
      <td>2.58</td>
      <td>3.55</td>
      <td>2.74</td>
    </tr>
    <tr>
      <td><a href="https://klingai.com">Kling-1.6 Pro Mode</a></td>
      <td>3.4</td>
      <td>3.56</td>
      <td>3.03</td>
      <td>3.58</td>
      <td>3.41</td>
    </tr>
    <tr>
      <td><a href="https://runwayml.com/research/introducing-runway-gen-4">Runway-Gen4</a></td>
      <td>3.39</td>
      <td>3.75</td>
      <td>3.2</td>
      <td>3.4</td>
      <td>3.37</td>
    </tr>
    <tr>
      <td>SkyReels-V2-DF</td>
      <td>3.24</td>
      <td>3.64</td>
      <td>3.21</td>
      <td>3.18</td>
      <td>2.93</td>
    </tr>
    <tr>
      <td>SkyReels-V2-I2V</td>
      <td>3.29</td>
      <td>3.42</td>
      <td>3.18</td>
      <td>3.56</td>
      <td>3.01</td>
    </tr>
  </tbody>
</table>
</p>

Our results demonstrate that both **SkyReels-V2-I2V (3.29)** and **SkyReels-V2-DF (3.24)** achieve state-of-the-art performance among open-source models, significantly outperforming HunyuanVideo-13B (2.84) and Wan2.1-14B (2.85) across all quality dimensions. With an average score of 3.29, SkyReels-V2-I2V demonstrates comparable performance to proprietary models Kling-1.6 (3.4) and Runway-Gen4 (3.39).


#### VBench
To objectively compare SkyReels-V2 T2V Model against other leading open-source Text-To-Video models, we conduct comprehensive evaluations using the public benchmark <a href="https://github.com/Vchitect/VBench">V-Bench</a>. Our evaluation specifically leverages the benchmark’s longer version prompt. For fair comparison with baseline models, we strictly follow their recommended setting for inference. 

<p align="center">
<table align="center">
  <thead>
    <tr>
      <th>Model</th>
      <th>Total Score</th>
      <th>Quality Score</th>
      <th>Semantic Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/hpcaitech/Open-Sora">OpenSora 2.0</a></td>
      <td>81.5 %</td>
      <td>82.1 %</td>
      <td>78.2 %</td>
    </tr>
    <tr>
      <td><a href="https://github.com/THUDM/CogVideo">CogVideoX1.5-5B</a></td>
      <td>80.3 %</td>
      <td>80.9 %</td>
      <td>77.9 %</td>
    </tr>
    <tr>
      <td><a href="https://github.com/Tencent/HunyuanVideo">HunyuanVideo-13B</a></td>
      <td>82.7 %</td>
      <td>84.4 %</td>
      <td>76.2 %</td>
    </tr>
    <tr>
      <td><a href="https://github.com/Wan-Video/Wan2.1">Wan2.1-14B</a></td>
      <td>83.7 %</td>
      <td>84.2 %</td>
      <td><strong>81.4 %</strong></td>
    </tr>
    <tr>
      <td>SkyReels-V2</td>
      <td><strong>83.9 %</strong></td>
      <td><strong>84.7 %</strong></td>
      <td>80.8 %</td>
    </tr>
  </tbody>
</table>
</p>

The VBench results demonstrate that SkyReels-V2 outperforms all compared models including HunyuanVideo-13B and Wan2.1-14B, With the highest **total score (83.9%)** and **quality score (84.7%)**. In this evaluation, the semantic score is slightly lower than Wan2.1-14B, while we outperform Wan2.1-14B in human evaluations, with the primary gap attributed to V-Bench’s insufficient evaluation of shot-scenario semantic adherence.
 
## Acknowledgements
We would like to thank the contributors of <a href="https://github.com/Wan-Video/Wan2.1">Wan 2.1</a>, <a href="https://github.com/xdit-project/xDiT">XDit</a> and <a href="https://qwenlm.github.io/blog/qwen2.5/">Qwen 2.5</a> repositories, for their open research and contributions.

## Citation

```bibtex
@misc{chen2025skyreelsv2infinitelengthfilmgenerative,
      title={SkyReels-V2: Infinite-length Film Generative Model}, 
      author={Guibin Chen and Dixuan Lin and Jiangping Yang and Chunze Lin and Juncheng Zhu and Mingyuan Fan and Hao Zhang and Sheng Chen and Zheng Chen and Chengchen Ma and Weiming Xiong and Wei Wang and Nuo Pang and Kang Kang and Zhiheng Xu and Yuzhe Jin and Yupeng Liang and Yubing Song and Peng Zhao and Boyuan Xu and Di Qiu and Debang Li and Zhengcong Fei and Yang Li and Yahui Zhou},
      year={2025},
      eprint={2504.13074},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.13074}, 
}
```
