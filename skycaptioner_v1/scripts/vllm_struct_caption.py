
import torch
import decord
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor

from torch.utils.data import DataLoader

SYSTEM_PROMPT = "I need you to generate a structured and detailed caption for the provided video. The structured output and the requirements for each field are as shown in the following JSON content: {\"subjects\": [{\"appearance\": \"Main subject appearance description\", \"action\": \"Main subject action\", \"expression\": \"Main subject expression  (Only for human/animal categories, empty otherwise)\", \"position\": \"Subject position in the video (Can be relative position to other objects or spatial description)\", \"TYPES\": {\"type\": \"Main category (e.g., Human)\", \"sub_type\": \"Sub-category (e.g., Man)\"}, \"is_main_subject\": true}, {\"appearance\": \"Non-main subject appearance description\", \"action\": \"Non-main subject action\", \"expression\": \"Non-main subject expression (Only for human/animal categories, empty otherwise)\", \"position\": \"Position of non-main subject 1\", \"TYPES\": {\"type\": \"Main category (e.g., Vehicles)\", \"sub_type\": \"Sub-category (e.g., Ship)\"}, \"is_main_subject\": false}], \"shot_type\": \"Shot type(Options: long_shot/full_shot/medium_shot/close_up/extreme_close_up/other)\", \"shot_angle\": \"Camera angle(Options: eye_level/high_angle/low_angle/other)\", \"shot_position\": \"Camera position(Options: front_view/back_view/side_view/over_the_shoulder/overhead_view/point_of_view/aerial_view/overlooking_view/other)\", \"camera_motion\": \"Camera movement description\", \"environment\": \"Video background/environment description\", \"lighting\": \"Lighting information in the video\"}"


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, model_path):
        self.meta = pd.read_csv(csv_path)
        self._path = 'path'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
  
    def __getitem__(self, index):
        row = self.meta.iloc[index]
        path = row[self._path]
        real_index = self.meta.index[index]
        vr = decord.VideoReader(path, ctx=decord.cpu(0), width=360, height=420)
        start = 0
        end = len(vr)
        # avg_fps = vr.get_avg_fps()
        index = self.get_index(end-start, 16, st=start)
        frames = vr.get_batch(index).asnumpy() # n h w c
        video_inputs = [torch.from_numpy(frames).permute(0, 3, 1, 2)]
        conversation = {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": row['path'],
                            "max_pixels": 360 * 420, # 460800
                            "fps": 2.0,
                        },
                        {   
                            "type": "text", 
                            "text": SYSTEM_PROMPT
                        },
                    ],
                }
                
        # 生成 user_input
        user_input = self.processor.apply_chat_template(
            [conversation],
            tokenize=False,
            add_generation_prompt=True
        )
        results = dict()
        inputs = {
            'prompt': user_input,
            'multi_modal_data': {'video': video_inputs}
        }
        results["index"] = real_index
        results['input'] = inputs
        return results

    def __len__(self):
        return len(self.meta)

    def get_index(self, video_size, num_frames, st=0):
        seg_size = max(0., float(video_size - 1) / num_frames)
        max_frame = int(video_size) - 1
        seq = []
        # index from 1, must add 1
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            # end = int(np.round(seg_size * (i + 1)))
            idx = min(start, max_frame)
            seq.append(idx+st)
        return seq
    
def result_writer(indices_list: list, result_list: list, meta: pd.DataFrame, column):
    flat_indices = []
    for x in zip(indices_list):
        flat_indices.extend(x)
    flat_results = []
    for x in zip(result_list):
        flat_results.extend(x)
    
    flat_indices = np.array(flat_indices)
    flat_results = np.array(flat_results)

    unique_indices, unique_indices_idx = np.unique(flat_indices, return_index=True)
    meta.loc[unique_indices, column[0]] = flat_results[unique_indices_idx]

    meta = meta.loc[unique_indices]
    return meta


def worker_init_fn(worker_id):
    # Set different seed for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    # Prevent deadlocks by setting timeout
    torch.set_num_threads(1)

def main():
    parser = argparse.ArgumentParser(description="SkyCaptioner-V1 vllm batch inference")
    parser.add_argument("--input_csv", default="./examples/test.csv")
    parser.add_argument("--out_csv", default="./examples/test_result.csv")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--model_path", required=True, type=str, help="skycaptioner-v1 model path")
    args = parser.parse_args()
    
    dataset = VideoTextDataset(csv_path=args.input_csv, model_path=args.model_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        timeout=180,
    )

    sampling_params = SamplingParams(temperature=0.05, max_tokens=2048)
    
    llm = LLM(model=args.model_path,
        gpu_memory_utilization=0.6, 
        max_model_len=31920,
        tensor_parallel_size=args.tp)
    
    indices_list = []
    caption_save = []
    for video_batch in tqdm(dataloader):
        indices = video_batch["index"]
        inputs = video_batch["input"]
        batch_user_inputs = []
        for prompt, video in zip(inputs['prompt'], inputs['multi_modal_data']['video'][0]):
            usi={'prompt':prompt, 'multi_modal_data':{'video':video}}
            batch_user_inputs.append(usi)
        outputs = llm.generate(batch_user_inputs, sampling_params, use_tqdm=False)
        struct_outputs = [output.outputs[0].text for output in outputs]

        indices_list.extend(indices.tolist())
        caption_save.extend(struct_outputs)
    
    meta_new = result_writer(indices_list, caption_save, dataset.meta, column=["structural_caption"])
    meta_new.to_csv(args.out_csv, index=False)
    print(f'Saved structural_caption to {args.out_csv}')

if __name__ == '__main__':
    main()