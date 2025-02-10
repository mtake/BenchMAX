from datasets import load_dataset
import random

from diff_match_patch import diff_match_patch

def load_testset(task_name, lang, is_src=False):
    match task_name:
        case "ifeval":
            testset = load_dataset("LLaMAX/BenchMAX_Domain_Translation", name=f"ifeval_{lang}", split="train")
        case "gpqa":
            testset = load_dataset("LLaMAX/BenchMAX_Domain_Translation", name=f"gpqa_{lang}", split="train")
        case "lcb_v4":
            testset = load_dataset("LLaMAX/BenchMAX_Domain_Translation", name=f"lcb_v4_{lang}", split="train")
        case "mgsm":
            testset = load_dataset("LLaMAX/BenchMAX_Domain_Translation", name=f"mgsm_{lang}", split="train")
        case "humaneval":
            testset = load_dataset("LLaMAX/BenchMAX_Domain_Translation", name=f"humaneval_{lang}", split="train")
        case "nexus":
            testset = load_dataset("LLaMAX/BenchMAX_Domain_Translation", name=f"nexus_{lang}", split="train")
        case "arenahard":
            testset = load_dataset("LLaMAX/BenchMAX_Domain_Translation", name=f"arenahard_{lang}", split="train")
        case "flores":
            testset = load_dataset("LLaMAX/BenchMAX_General_Translation", name=f"flores_{lang}", split="train")
        case _:
            raise ValueError(f"Unknown task: {task_name}")
    if is_src and type(testset[0]["text"]) is list:
        testset = testset.map(lambda x: {"text": x["text"][0]})
    return testset

def load_unidirectional_testset(task_name, src_lang, tgt_lang):
    match task_name:
        case "wmt24":
            src_dataset = load_dataset("LLaMAX/BenchMAX_General_Translation", name=f"wmt24_{src_lang}-{tgt_lang}_{src_lang}", split="train")
            tgt_dataset = load_dataset("LLaMAX/BenchMAX_General_Translation", name=f"wmt24_{src_lang}-{tgt_lang}_{tgt_lang}", split="train")
        case "ted":
            src_dataset = load_dataset("LLaMAX/BenchMAX_General_Translation", name=f"ted_{src_lang}", split="train")
            tgt_dataset = load_dataset("LLaMAX/BenchMAX_General_Translation", name=f"ted_{tgt_lang}", split="train")
            src_dataset, tgt_dataset = process_ted_data(src_dataset, tgt_dataset)
        case _:
            raise ValueError(f"Unknown task: {task_name}")
    return src_dataset, tgt_dataset

def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg

def post_process(outputs, task):
    match task:
        case "flores" | "ted" | "wmt24":
            for i in range(len(outputs)):
                outputs[i] = outputs[i].strip().split("\n")[0]
        case "ifeval" | "arenahard":
            for i in range(len(outputs)):
                if outputs[i].startswith('"""'):
                    outputs[i] = outputs[i].removeprefix('"""').lstrip()
                if outputs[i].endswith('"""'):
                    outputs[i] = outputs[i].removesuffix('"""').rstrip()
        case "humaneval":
            for i in range(len(outputs)):
                if outputs[i].startswith("```python"):
                    outputs[i] = outputs[i].removeprefix("```python").lstrip()
                if outputs[i].endswith("```"):
                    outputs[i] = outputs[i].removesuffix("```").rstrip()
        case _:
            pass

def get_directions(src_lang, tgt_lang, langs):
    directions = []
    if langs is not None:
        langs = langs.split(",")
        for src_lang in langs:
            for tgt_lang in langs:
                if src_lang == tgt_lang:
                    continue
                directions.append((src_lang, tgt_lang))
    else:
        src_lang = src_lang.split(",")
        tgt_lang = tgt_lang.split(",")
        for s_l in src_lang:
            for t_l in tgt_lang:
                if s_l == t_l:
                    continue
                directions.append((s_l, t_l))
    return directions


def is_unidirectional(task_name):
    return task_name in ["wmt24", "ted"]


def process_ted_data(src_dataset, tgt_dataset):
    new_src_dataset = src_dataset.filter(lambda x: "__NULL__" not in x["text"] and "_ _ NULL _ _" not in x["text"])
    new_tgt_dataset = tgt_dataset.filter(lambda x: "__NULL__" not in x["text"] and "_ _ NULL _ _" not in x["text"])
    
    src_ids = set(new_src_dataset["id"])
    tgt_ids = set(new_tgt_dataset["id"])
    common_ids = src_ids.intersection(tgt_ids)
    if len(common_ids) > 1000:
        random.seed(42)
        common_ids = random.sample(common_ids, 1000)
    
    new_src_dataset = new_src_dataset.filter(lambda x: x["id"] in common_ids)
    new_tgt_dataset = new_tgt_dataset.filter(lambda x: x["id"] in common_ids)
    new_src_dataset.remove_columns(["id", "talk_name", "lang"])
    new_tgt_dataset.remove_columns(["id", "talk_name", "lang"])
    return new_src_dataset, new_tgt_dataset

def convert_to_refs(tgt_datasets):
    for k, v in tgt_datasets.items():
        if type(v["text"][0]) is list:
            tgt_datasets[k] = [ref for ref in zip(*v["text"])]
        else:
            tgt_datasets[k] = [v["text"]]


def diff_lineMode(text1, text2):
    dmp = diff_match_patch()
    a = dmp.diff_linesToChars(text1, text2)
    lineText1 = a[0]
    lineText2 = a[1]
    lineArray = a[2]
    diffs = dmp.diff_main(lineText1, lineText2, False)
    dmp.diff_charsToLines(diffs, lineArray)
    return diffs
