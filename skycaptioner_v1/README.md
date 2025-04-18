# SkyCaptioner-V1: A Structured Video Captioning Model

<p align="center">
🤗 <a href="https://huggingface.co/Skywork/SkyCaptioner-V1" target="_blank">Hugging Face</a> · 💬 <a href="https://discord.gg/PwM6NYtccQ" target="_blank">Discord</a>
</p>

---

Welcome to the SkyCaptioner-V1 repository! Here, you'll find the structural caption model weights and inference code for our video captioner that labels the video data efficiently and comprehensively.

## 🔥🔥🔥 News!!

* Apr 21, 2025: 👋 We release the [vllm](https://github.com/vllm-project/vllm) batch inference code for SkyCaptioner-V1  Model.
* Apr 21, 2025: 👋 We release the first film-video-oriented caption model [SkyCaptioner-V1  Model](https://huggingface.co/Skywork/SkyCaptioner-V1). For more details, please check our [paper](https://arxiv.org/pdf/2504.13074).

## 📑 TODO List

- SkyCaptioner-V1
  
  - [x] Checkpoints
  - [x] Batch Inference Code
  - [x] Caption Fusion Method
  - [ ] Web Demo (Gradio)

## 🌟 Overview

SkyCaptioner-V1 is a unified video captioning model designed to generate high-quality, structured descriptions for video data. It integrates specialized sub-expert models and multimodal large language models (MLLMs) to address the limitations of general captioners in capturing professional film-related details. Key aspects include:

1. ​​**Structural Representation**​: Combines general video descriptions (from MLLMs) with expert-level annotations (e.g., shot types, camera motions).
2. ​​**Knowledge Distillation**​: Distills expertise from sub-expert captioners into a unified model.
3. ​​**Application Flexibility**​: Generates dense captions for text-to-video (T2V) and concise prompts for image-to-video (I2V) tasks.

## 🔑 Key Features

### Structured Caption Framework

Our Captioning model captures multi-dimensional details:

* ​​**Subjects**​: Appearance, action, expression, position, and hierarchical categorization.
* ​​**Shot Metadata**​: Shot type (e.g., close-up, long shot), shot angle, shot position, camera motion.
* **General desc**: environment, and lighting.

### Sub-Expert Integration

* ​​**Shot Captioner**​: Classifies shot type, angle, and position with high precision.
* ​​**Expression Captioner**​: Analyzes facial expressions, emotion intensity, and temporal dynamics.
* ​​**Camera Motion Captioner**​: Tracks 6DoF camera movements and composite motion types,

### Training Pipeline

* Trained on \~2M high-quality, concept-balanced videos curated from 10M raw samples.
* Fine-tuned on Qwen2.5-VL-7B-Instruct with a global batch size of 512 across 32 A800 GPUs.
* Optimized using AdamW (learning rate: 1e-5) for 2 epochs.

### Dynamic Caption Fusion:

* Adapts output length based on application (T2V/I2V).
* Employs LLM Model to fusion structural fields to get a natural and fluency caption for downstream tasks.

## 📊 Benchmark Results

SkyCaptioner-V1 demonstrates significant improvements over existing models in key film-specific captioning tasks, particularly in ​**shot-language understanding** and ​​**domain-specific precision**​. The differences stem from its structural architecture and expert-guided training:

1. ​​**Superior Shot-Language Understanding**​:
   * ​Our Captioner model outperforms Qwen2.5-VL-72B with +11.2% in shot type, +16.1% in shot angle, and +50.4% in shot position accuracy. Because SkyCaptioner-V1’s specialized shot classifiers outperform generalist MLLMs, which lack film-domain fine-tuning.
   * ​+28.5% accuracy in camera motion vs. Tarsier2-recap-7B (88.8% vs. 41.5%):
     Its 6DoF motion analysis and active learning pipeline address ambiguities in composite motions (e.g., tracking + panning) that challenge generic captioners.
2. ​​**High domain-specific precision**​:
   * ​​Expression accuracy​: ​68.8% vs. 54.3% (Tarsier2-recap-7B), leveraging temporal-aware S2D frameworks to capture dynamic facial changes.

<p align="center">
<table align="center">
  <thead>
    <tr>
      <th>Metric</th>
      <th>Qwen2.5-VL-7B-Ins.</th>
      <th>Qwen2.5-VL-72B-Ins.</th>
      <th>Tarsier2-recap-7B</th>
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

## 📦 Model Downloads

Our SkyCaptioner-V1 model can be downloaded from  [SkyCaptioner-V1  Model](https://huggingface.co/Skywork/SkyCaptioner-V1).
We use [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) as our caption fusion model to intelligently combine structured caption fields, producing either dense or sparse final captions depending on application requirements.

```shell
# download SkyCaptioner-V1
huggingface-cli download Skywork/SkyCaptioner-V1 /path/to/your_local_model_path
# download Qwen2.5-32B-Instruct
huggingface-cli download Qwen/Qwen2.5-32B-Instruct /path/to/your_local_model_path2
```

## 🛠️ Running Guide

Begin by cloning the repository:

```shell
git clone https://github.com/SkyworkAI/SkyReels-V2
cd skycaptioner_v1
```

### Installation Guide for Linux

We recommend Python 3.10 and CUDA version 12.2 for the manual installation.

```shell
pip install -r requirements.txt
```

### Running Command

#### Get Structural Caption by SkyCaptioner-V1

```shell
export SkyCaptioner_V1_Model_PATH="/path/to/your_local_model_path"

python scripts/vllm_struct_caption.py \
    --model_path ${SkyCaptioner_V1_Model_PATH} \
    --input_csv "./examples/test.csv" \
    --out_csv "./examepls/test_result.csv" \
    --tp 1 \
    --bs 4
```

#### T2V/I2V Caption Fusion by Qwen2.5-32B-Instruct Model

```shell
export LLM_MODEL_PATH="/path/to/your_local_model_path2"

python scripts/vllm_fusion_caption.py \
    --model_path ${LLM_MODEL_PATH} \
    --input_csv "./examples/test_result.csv" \
    --out_csv "./examples/test_result_caption.csv" \
    --bs 4 \
    --tp 1 \
    --task t2v
```
> **Note**: 
> - If you want to get i2v caption, just change the `--task t2v` to `--task i2v` in your Command.

## Acknowledgements

We would like to thank the contributors of <a href="https://github.com/QwenLM/Qwen2.5-VL">Qwen2.5-VL</a>, <a href="https://github.com/bytedance/tarsier">tarsier2</a> and <a href="https://github.com/vllm-project/vllm">vllm</a> repositories, for their open research and contributions.

## Citation

```bibtex
@misc{SkyReelsV2,
author = {Guibin Chen and Dixuan Lin and Jiangping Yang and Chunze Lin and Juncheng Zhu and Mingyuan Fan and Hao Zhang and Sheng Chen and Zheng Chen and Chengchen Ma and Weiming Xiong and Wei Wang and Nuo Pang and Kang Kang and Zhiheng Xu and Yuzhe Jin and Yupeng Liang and Yubing Song and Peng Zhao and Boyuan Xu and Di Qiu and Debang Li and Zhengcong Fei and Yang Li and Yahui Zhou},
title = {Skyreels V2:Infinite-Length Film Generative Model},
year = {2025},
eprint={2504.13074},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/2504.13074}
}
```


