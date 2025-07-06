#!/usr/bin/env bash

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

LANGS=()
LANGS+=("en,ja")
# LANGS+=("en")
# LANGS+=("ja")
MODELS=()
# MODELS+=("ibm-granite/granite-3.3-8b-instruct")
# MODELS+=("granite-3.3-8b-instruct-teigaku-genzei-interp")
# MODELS+=("granite-3.3-8b-instruct-teigaku-genzei")
MODELS+=("granite-3.3-8b-instruct-ibm-newsroom-d5-x100-interp")
MODELS+=("granite-3.3-8b-instruct-ibm-newsroom-d5-x100")
TASKS=()
TASKS+=("xgpqa")

ENV=""
#ENV="TOKENIZERS_PARALLELISM=false ${ENV}"
ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${ENV}"

for l in "${LANGS[@]}"; do
    for m in "${MODELS[@]}"; do
	for t in "${TASKS[@]}"; do
	    THIS_START_TIME="$(${DATE_CMD} +%s)"
	    THIS_START_TIME_STR="$(${DATE_CMD} -d @${THIS_START_TIME} +%Y%m%d-%H%M%S)"
	    echo "XXX THIS_DATETIME ${THIS_START_TIME_STR}" | tee -a ${LOGFILE}

	    cmd="${ENV}./run.sh ${m} ${t} ${l}"
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
