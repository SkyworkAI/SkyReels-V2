expor LLM_MODEL_PATH="/path/to/your_local_model_path2"

python scripts/vllm_fusion_caption.py \
    --model_path ${LLM_MODEL_PATH} \
    --input_csv "./examples/test_result.csv" \
    --out_csv "./examples/test_result_caption.csv" \
    --bs 4 \
    --tp 1 \
    --task t2v
