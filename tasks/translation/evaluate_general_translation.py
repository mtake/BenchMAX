import argparse
import os
import json

from datasets import concatenate_datasets

from utils import (
    load_testset,
    load_unidirectional_testset,
    get_directions,
    is_unidirectional,
    convert_to_refs,
)
from metrics import get_spBLEU, get_BLEU, get_chrf, get_ter, get_xComet

direction_task_dict = {
    "en-zh": ["ted", "wmt24"],
    "en-ar": ["ted"],
    "en-bn": ["ted"],
    "en-cs": ["ted", "wmt24"],
    "en-de": ["ted", "wmt24"],
    "en-es": ["ted"],
    "en-fr": ["ted"],
    "en-hu": ["ted"],
    "en-ja": ["ted", "wmt24"],
    "en-ko": ["ted"],
    "en-ru": ["ted", "wmt24"],
    "en-sr": ["ted"],
    "en-th": ["ted"],
    "en-vi": ["ted"],
    "en-sw": [],
    "en-te": [],
    "zh-en": ["ted"],
    "ar-en": ["ted"],
    "bn-en": ["ted"],
    "cs-en": ["ted"],
    "de-en": ["ted"],
    "es-en": ["ted"],
    "fr-en": ["ted"],
    "hu-en": ["ted"],
    "ja-en": ["ted"],
    "ko-en": ["ted"],
    "ru-en": ["ted"],
    "sr-en": ["ted"],
    "th-en": ["ted"],
    "vi-en": ["ted"],
    "sw-en": [],
    "te-en": [],
}
for v in direction_task_dict.values():
    v.insert(0, "flores")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-lang", "-s", type=str, default=None)
    parser.add_argument("--tgt-lang", "-t", type=str, default=None)
    parser.add_argument("--languages", "-l", type=str, default=None)
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

    directions = get_directions(src_lang, tgt_lang, langs)

    src_datasets, tgt_datasets = {}, {}
    for s_l, t_l in directions:
        d = f"{s_l}-{t_l}"
        for task_name in direction_task_dict[d]:
            if is_unidirectional(task_name):
                src_dataset, tgt_dataset = load_unidirectional_testset(task_name, s_l, t_l)
            else:
                src_dataset = load_testset(task_name, s_l, is_src=True)
                tgt_dataset = load_testset(task_name, t_l)
            if src_datasets.get(d, None) is None:
                src_datasets[d] = src_dataset
            else:
                src_datasets[d] = concatenate_datasets([src_datasets[d], src_dataset])
            if tgt_datasets.get(d, None) is None:
                tgt_datasets[d] = tgt_dataset
            else:
                tgt_datasets[d] = concatenate_datasets([tgt_datasets[d], tgt_dataset])
    convert_to_refs(tgt_datasets)

    model_name = os.path.basename(args.model_name)
    output_dir = os.path.join(args.output_dir, model_name, "general_translation")
    os.makedirs(output_dir, exist_ok=True)
    metrics = args.metrics.split(",")
    for s_l, t_l in directions:
        hyps = []
        for task_name in direction_task_dict[f"{s_l}-{t_l}"]:
            prev_result_file = f"{args.output_dir}/{model_name}/{task_name}/result_{s_l}-{t_l}.json"
            with open(prev_result_file, "r") as f:
                prev_results = json.load(f)
            hyps.extend(prev_results["outputs"])
        refs = tgt_datasets[f"{s_l}-{t_l}"]

        results = {}
        result_file = f"{output_dir}/result_{s_l}-{t_l}.json"
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                results = json.load(f)
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
                    srcs = src_datasets[f"{s_l}-{t_l}"]["text"]
                    score = get_xComet(srcs, hyps, refs)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                results[metric] = score
            else:
                score = results[metric]
        print(f"{model_name} {s_l} -> {t_l} {metric} score: {score}")
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()