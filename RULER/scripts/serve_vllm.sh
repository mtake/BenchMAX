set -e
if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi
export VLLM_USE_MODELSCOPE=False
# export HF_HUB_CACHE=/cpfs01/shared/XNLP_H800/hf_hub
# export HF_HUB_OFFLINE=True

# Root Directories
GPUS="4" # GPU size for tensor_parallel.
MODEL_DIR="/cpfs01/shared/XNLP_H800/hf_hub" # the path that contains individual model folders from HUggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.

# Model and Tokenizer
source config_models.sh
MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi


export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}

if [ "$MODEL_FRAMEWORK" == "vllm" ]; then
    python pred/serve_vllm.py \
        --model=${MODEL_PATH} \
        --tensor-parallel-size=${GPUS} \
        --dtype bfloat16 \
        --disable-custom-all-reduce \
        --gpu-memory-utilization 0.95 \
        --enforce-eager \
        --max-model-len 131072 \
        --max-num-batched-tokens 262144
fi