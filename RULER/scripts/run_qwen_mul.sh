set -e
for lang in en zh es fr de ru ja th sw bn te ar ko vi cs hu sr; do
    bash run.sh qwen2.5-72b-chat synthetic $lang
done
