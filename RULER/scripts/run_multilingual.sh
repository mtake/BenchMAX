set -e
for lang in en zh es fr de ru ja th sw bn te ar ko vi cs hu sr; do
# for lang in cs hu sr; do
    bash run.sh llama3.1-70b-chat synthetic $lang
done