import argparse
import os
import json

from utils import (
    load_testset,
    load_unidirectional_testset,
    get_directions,
    is_unidirectional,
    convert_to_refs,
)
from metrics import get_spBLEU, get_BLEU, get_chrf, get_ter, get_xComet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-lang", "-s", type=str, default=None)
    parser.add_argument("--tgt-lang", "-t", type=str, default=None)
    parser.add_argument("--languages", "-l", type=str, default=None)
    parser.add_argument("--task-name", type=str, required=True, help="The task name of test set")
    parser.add_argument("--metrics", type=str, help="The metrics to evaluate the translations, splitted by comma. Available metrics: spBLEU, ChrF, TER")
    parser.add_argument("--model-name", "-m", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output")
    args = parser.parse_args()
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    langs = args.languages
    if langs is None and (src_lang is None or tgt_lang is None):
        raise ValueError("Either --languages or --src-lang and --tgt-lang should be provided")
    if langs is not None and (src_lang is not None or tgt_lang is not None):
        raise ValueError("Only one of --languages or --src-lang and --tgt-lang should be provided")

    task_name = args.task_name
    directions = get_directions(src_lang, tgt_lang, langs)
    unidirectional = is_unidirectional(task_name)

    print("Loading test set and building prompts...")
    src_datasets, tgt_datasets = {}, {}
    if not unidirectional:
        for s_l, t_l in directions:
            if s_l not in src_datasets:
                src_datasets[s_l] = load_testset(task_name, s_l, is_src=True)
            if t_l not in tgt_datasets:
                tgt_datasets[t_l] = load_testset(task_name, t_l)
    else:
        for s_l, t_l in directions:
            src_dataset, tgt_dataset = load_unidirectional_testset(task_name, s_l, t_l)
            d = f"{s_l}-{t_l}"
            src_datasets[d] = src_dataset
            tgt_datasets[d] = tgt_dataset
    convert_to_refs(tgt_datasets)

    model_name = os.path.basename(args.model_name)
    output_dir = os.path.join(args.output_dir, model_name, task_name)
    metrics = args.metrics.split(",")
    for s_l, t_l in directions:
        result_file = f"{output_dir}/result_{s_l}-{t_l}.json"
        with open(result_file, "r") as f:
            results = json.load(f)
        hyps = results["outputs"]
        refs = tgt_datasets[f"{s_l}-{t_l}"] if unidirectional else tgt_datasets[t_l]
        for metric in metrics:
            if metric not in results:
                if metric == "spBLEU":
                    score = get_spBLEU(hyps, refs)
                elif metric == "BLEU":
                    score = get_BLEU(hyps, refs, t_l)
                elif metric == "ChrF":
                    score = get_chrf(hyps, refs)
                elif metric == "TER":
                    score = get_ter(hyps, refs)
                elif metric == "xComet":
                    srcs = src_datasets[s_l]["text"]
                    score = get_xComet(srcs, hyps, refs)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                results[metric] = score
            else:
                score = results[metric]
            print(f"{task_name} {s_l} -> {t_l} {metric} score: {score}")
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()