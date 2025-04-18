expor SkyCaptioner_V1_Model_PATH="/path/to/your/model_path"

python scripts/vllm_inference.py \
    --model_path ${SkyCaptioner_V1_Model_PATH} \
    --input_csv "./examples/test.csv" \
    --out_csv "./examepls/test_result.csv" \
    --tp 1 \
    --bs 32