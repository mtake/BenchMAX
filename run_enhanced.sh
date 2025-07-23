#!/bin/bash
# run.sh: A unified script to run different tasks in BenchMAX.
#
# Usage:
#   ./run.sh <model> <task> <languages> [additional arguments]
#
# Arguments:
#   model    : The model name or path to be evaluated.
#   task     : The task to run. Available tasks include:
#              - rule_based (or xifeval)
#              - model_based (or xarenahard)
#              - function-completion (or xhumaneval)
#              - problem-solving (or xlcb)
#              - math (or xmgsm)
#              - science (or xgpqa)
#              - question-answering (or xruler)
#   languages : The language code to evaluate (e.g., en, zh, etc.). For some tasks,
#              use "all" if a multi-language evaluation is supported.
#
# Optional:
#   Other arguments may be needed for some tasks (e.g., for problem-solving tasks you should
#   specify the local_model_path as the 4th parameter).
#
# Ensure that the necessary repositories (e.g., lm-evaluation-harness, arena-hard-auto, etc.)
# are cloned/installed as detailed in the BenchMAX README.
#
# Example:
#   ./run.sh meta-llama/Llama-3.1-8B-Instruct rule_based en
#   ./run.sh /path/to/your/local/model xgpqa all
#   ./run.sh meta-llama/Llama-3.1-8B-Instruct problem-solving all /path/to/your/local/model

if [ $# -lt 3 ]; then
    echo "Usage: $0 <model> <task> <languages> [additional arguments]"
    exit 1
fi

MODEL=$1
TASK=$2
LANG=$3

echo "Model: $MODEL"
echo "Task: $TASK"
echo "Languages: $LANG"

if [ "$LANG" == "all" ]; then
    LANGS=(en ar bn cs de es fr hu ja ko ru sr sw te th vi zh)
else
    IFS=',' read -r -a LANGS <<< "$LANG"
fi

TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
echo "XXX TENSOR_PARALLEL_SIZE: ${TENSOR_PARALLEL_SIZE}"

VLLM_OPT=""
if (( TENSOR_PARALLEL_SIZE > 1 )); then
    VLLM_OPT=",tensor_parallel_size=${TENSOR_PARALLEL_SIZE}${VLLM_OPT}"
    VLLM_OPT=",disable_custom_all_reduce=True${VLLM_OPT}"
fi

case "$TASK" in
    # --------------------------- Rule-based Instruction Following Task ---------------------------
    "rule_based"|"xifeval")
        if [ "$LANG" == "all" ]; then
            echo "Running Rule-based Instruction Following Task on all languages..."
            lm-eval -m vllm --model_args pretrained=${MODEL}${VLLM_OPT} --tasks xifeval_multi --batch_size auto --apply_chat_template --include_path tasks/ifeval --log_samples -o results
        else
            echo "Running Rule-based Instruction Following Task on language: ${LANGS[@]}..."
            tasks=$(IFS=','; echo "${LANGS[*]/#/xifeval_}")
            lm-eval -m vllm --model_args pretrained=${MODEL}${VLLM_OPT} --tasks $tasks --batch_size auto --apply_chat_template --include_path tasks/ifeval --log_samples -o results
        fi
        ;;
    # --------------------------- Model-based Instruction Following Task ---------------------------
    "model_based"|"xarenahard")
        echo "Running Model-based Instruction Following Task using Arena-hard for language: ${LANGS[@]}..."
        pushd tasks/arenahard > /dev/null
        # Prepare arena-hard environment (downloads, preprocessing, etc.)
        if [ ! -d "arena-hard-auto" ]; then
            bash prepare.sh
        fi
        echo "Generating responses for language: $LANGS..."
        for lang in "${LANGS[@]}"; do
            python gen_answer.py --setting-file config/gen_answer_config_${lang}.yaml
        done
        echo "Running LLM-as-a-judge for language: $LANGS..."
        for lang in "${LANGS[@]}"; do
            python gen_judgment.py --setting-file config/judge_config_${lang}.yaml
        done
        popd > /dev/null
        ;;
    # --------------------------- Function Completion Task ---------------------------
    "function-completion"|"xhumaneval")
        pushd tasks/evalplus > /dev/null
        echo "Running Function Completion Task using evalplus for language: ${LANGS[@]}..."
        for lang in "${LANGS[@]}"; do
            python -m evalplus.evaluate --model ${MODEL} --dataset humaneval --backend vllm --greedy --lang ${lang}
        done
        popd > /dev/null
        ;;
    # --------------------------- Programming Problem Solving Task ---------------------------
    "problem-solving"|"xlcb")
        local_model_path=$4
        pushd tasks/LiveCodeBench > /dev/null
        if [ -z "$local_model_path" ]; then
            echo "Warning: The fourth argument local_model_path is not set. It will use the model from huggingface. Please set it to the path of your local model if you want to use the local model."
        fi
        echo "Running Programming Problem Solving Task using LiveCodeBench for language: ${LANGS[@]}..."
        for lang in "${LANGS[@]}"; do
            if [ -z "$local_model_path" ]; then
                python -m lcb_runner.runner.main --model ${MODEL} --release_version release_v4 --dataset ${lang} --evaluate --num_process_evaluate 16
            else
                python -m lcb_runner.runner.main --model ${MODEL} --local_model_path ${local_model_path} --release_version release_v4 --dataset ${lang} --evaluate --num_process_evaluate 16
            fi
        done
        popd > /dev/null
        ;;
    # --------------------------- Math Reasoning Task ---------------------------
    "math"|"xmgsm")
        if [ "$LANG" == "all" ]; then
            echo "Running Math Reasoning Task on all languages..."
            lm-eval -m vllm --model_args pretrained=${MODEL}${VLLM_OPT} --tasks xmgsm_native_cot_multi --batch_size auto --apply_chat_template --include_path tasks/mgsm --log_samples -o results
        else
            echo "Running Math Reasoning Task on language: ${LANGS[@]}..."
            tasks=$(IFS=','; echo "${LANGS[*]/#/xmgsm_native_cot_}")
            lm-eval -m vllm --model_args pretrained=${MODEL}${VLLM_OPT} --tasks $tasks --batch_size auto --apply_chat_template --include_path tasks/mgsm --log_samples -o results
        fi
        ;;
    # --------------------------- Science Reasoning Task ---------------------------
    "science"|"xgpqa")
        if [ "$LANG" == "all" ]; then
            echo "Running Science Reasoning Task on all languages..."
            lm-eval -m vllm --model_args pretrained=${MODEL}${VLLM_OPT} --tasks xgpqa_main_native_cot_zeroshot_multi --batch_size auto --apply_chat_template --include_path tasks/gpqa --log_samples -o results
        else
            echo "Running Science Reasoning Task on language: ${LANGS[@]}..."
            tasks=$(IFS=','; echo "${LANGS[*]/#/xgpqa_main_native_cot_zeroshot_}")
            lm-eval -m vllm --model_args pretrained=${MODEL}${VLLM_OPT} --tasks $tasks --batch_size auto --apply_chat_template --include_path tasks/gpqa --log_samples -o results
        fi
        ;;
    # --------------------------- Long Context Task ---------------------------
    "question-answering"|"xruler")
        echo "Don't forget to prepare model path and data path in ruler's config scripts config_models.sh and run.sh"
        pushd tasks/RULER/scripts > /dev/null
        pushd data/synthetic/json > /dev/null
        if [ ! -d "haystack"]; then
            bash download_haystack.sh
        fi
        if [ ! -d "qas" ]; then
            bash download_qa_dataset.sh
        fi
        popd > /dev/null
        echo "Running Long Context Task using RULER for language: ${LANGS[@]}..."
        for lang in "${LANGS[@]}"; do
            bash run.sh ${MODEL} synthetic ${lang}
        done
        popd > /dev/null
        ;;
    # --------------------------- Tool Use Task ---------------------------
    "tool-use"|"xnexus")
        pushd tasks/nexus > /dev/null
        if [ "$MODEL" == *"Llama-3"* ]; then
            parser=llama3
        elif [ "$MODEL" == *"Qwen2"* ]; then
            parser=qwen2
        else
            parser=generic
        fi
        echo "Running Tool Use Task using Nexus for language: ${LANGS[@]}..."
        for lang in "${LANGS[@]}"; do
            python evaluator.py -m ${MODEL} --infer-backend vllm -t ${lang} --output-parser-name $parser
        done
        popd > /dev/null
        ;;
    # --------------------------- Unknown Task ---------------------------
    *)
        echo "Unknown task: $TASK"
        echo "Available tasks:"
        echo "  rule_based (or xifeval)"
        echo "  model_based (or xarenahard)"
        echo "  function-completion (or xhumaneval)"
        echo "  problem-solving (or xlcb)"
        echo "  math (or xmgsm)"
        echo "  science (or xgpqa)"
        echo "  question-answering (or xruler)"
        echo "  tool-use (or xnexus)"
        exit 1
        ;;
esac

echo "Task '$TASK' completed for model '${MODEL}' and language '${LANG}'."
