expor SkyCaptioner_V1_Model_PATH="/path/to/your_local_model_path"

python scripts/vllm_struct_caption.py \
    --model_path ${SkyCaptioner_V1_Model_PATH} \
    --input_csv "./examples/test.csv" \
    --out_csv "./examepls/test_result.csv" \
    --tp 1 \
    --bs 32