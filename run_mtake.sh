#!/usr/bin/env bash

#
# Run on a Linux machine with GPU
#

# for macOS
if command -v gdate &> /dev/null
then
    DATE_CMD=gdate
else
    DATE_CMD=date
fi

START_TIME="$(${DATE_CMD} +%s)"
START_TIME_STR="$(${DATE_CMD} -d @${START_TIME} +%Y%m%d-%H%M%S)"
BASENAME="$(basename "${BASH_SOURCE}" .sh)"
HOSTNAME_S="$(hostname -s)"
LOGFILE="${BASENAME}-${START_TIME_STR}-${HOSTNAME_S}.log"
echo "XXX LOGFILE ${LOGFILE}" | tee -a ${LOGFILE}
echo "XXX DATETIME ${START_TIME_STR}" | tee -a ${LOGFILE}

# count gpus
if command -v nvidia-smi >/dev/null 2>&1; then
    tensor_parallel_size=$(nvidia-smi --list-gpus | wc -l)
else
    tensor_parallel_size=0
fi
echo "XXX tensor_parallel_size: ${tensor_parallel_size}" | tee -a ${LOGFILE}

if (( tensor_parallel_size == 0 )); then
    echo "ERROR: A GPU is required to run this command. Exiting..." | tee -a ${LOGFILE}
    exit 1
fi

LANGS=()
LANGS+=("en,ja")
# LANGS+=("en")
# LANGS+=("ja")

MODELS=()
# MODELS+=("ibm-granite/granite-3.3-8b-instruct")
# MODELS+=("granite-3.3-8b-instruct-teigaku-genzei-interp")
# MODELS+=("granite-3.3-8b-instruct-teigaku-genzei")
# MODELS+=("granite-3.3-8b-instruct-ibm-newsroom-d5-x100-interp")
# MODELS+=("granite-3.3-8b-instruct-ibm-newsroom-d5-x100")
# MODELS+=("granite-3.3-8b-instruct-jfe-technical-report_r5-interp" "granite-3.3-8b-instruct-jfe-technical-report_r5")
# MODELS+=("granite-4.0-tiny-prerelease-greylock-r250721a")
MODELS+=("granite-4.0-small-prerelease-greylock-r250721a")

TASKS=()
TASKS+=("xgpqa")

ENV=""
#ENV="TOKENIZERS_PARALLELISM=false ${ENV}"
ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${ENV}"
#ENV="VLLM_WORKER_MULTIPROC_METHOD=spawn ${ENV}"  # @@@ahoaho XXX WIP
ENV="TENSOR_PARALLEL_SIZE=${tensor_parallel_size} ${ENV}"

for l in "${LANGS[@]}"; do
    for m in "${MODELS[@]}"; do
	for t in "${TASKS[@]}"; do
	    THIS_START_TIME="$(${DATE_CMD} +%s)"
	    THIS_START_TIME_STR="$(${DATE_CMD} -d @${THIS_START_TIME} +%Y%m%d-%H%M%S)"
	    echo "XXX THIS_DATETIME ${THIS_START_TIME_STR}" | tee -a ${LOGFILE}

	    # cmd="${ENV}./run.sh ${m} ${t} ${l}"
	    cmd="${ENV}./run_enhanced.sh ${m} ${t} ${l}"
	    echo "$cmd" | tee -a ${LOGFILE}
	    eval "$cmd" 2>&1 | tee -a ${LOGFILE}

	    THIS_END_TIME="$(${DATE_CMD} +%s)"
	    THIS_END_TIME_STR="$(${DATE_CMD} -d @${THIS_END_TIME} +%Y%m%d-%H%M%S)"
	    echo "XXX THIS_DATETIME ${THIS_END_TIME_STR}" | tee -a ${LOGFILE}
	    echo "XXX THIS_ELAPSED_SECS $((THIS_END_TIME - THIS_START_TIME))" | tee -a ${LOGFILE}
	done
    done
done

END_TIME="$(${DATE_CMD} +%s)"
END_TIME_STR="$(${DATE_CMD} -d @${END_TIME} +%Y%m%d-%H%M%S)"
echo "XXX DATETIME ${END_TIME_STR}" | tee -a ${LOGFILE}
echo "XXX ELAPSED_SECS $((END_TIME - START_TIME))" | tee -a ${LOGFILE}
