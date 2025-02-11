git clone https://github.com/lmarena/arena-hard-auto.git

cp -r data arena-hard-auto/
cp -r config arena-hard-auto/
mv arena-hard-auto/data/arena-hard-v0.1 arena-hard-auto/data/arena-hard-v0.1-en

git clone https://huggingface.co/datasets/LLaMAX/BenchMAX_Model-based
for lang in ar bn cs de es fr hu ja ko ru sr sw te th vi zh; do
    mv BenchMAX_Model-based/arenahard_${lang}.jsonl arena-hard-auto/data/arena-hard-v0.1-${lang}/question.jsonl
done
