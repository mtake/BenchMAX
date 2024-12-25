# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
import re
import json
import argparse
import importlib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) 
from tokenizer import select_tokenizer

def write_manifest(output_path, target_manifest, ensure_ascii=True):
    """
    Write to manifest file

    Args:
        output_path (str or Path): Path to output manifest file
        target_manifest (list): List of manifest file entries
        ensure_ascii (bool): default is True, meaning the output is guaranteed to have all incoming non-ASCII characters escaped. If ensure_ascii is false, these characters will be output as-is.
    """
    with open(output_path, "w", encoding="utf-8") as outfile:
        for tgt in target_manifest:
            json.dump(tgt, outfile, ensure_ascii=ensure_ascii)
            outfile.write('\n')


parser = argparse.ArgumentParser()
# Basic Configurations
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--save_name", type=str, required=True, help='name of the save dataset jsonl file')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--tokenizer_path", type=str, required=True, help='path to the tokenizer model')
parser.add_argument("--tokenizer_type",  type=str, default='nemo', help='[Options] nemo, hf, openai.')
parser.add_argument("--max_seq_length", type=int, required=True, help='max sequence length including all input tokens and generated tokens.')
parser.add_argument("--tokens_to_generate", type=int, required=True, help='expected generated token amount.')
parser.add_argument("--num_samples", type=int, required=True, help='number of samples to generate')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--template", type=str, default='', help='prompt template')
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')
parser.add_argument("--language", type=str, default='en', help='The language of the text')

# Complexity Configurations
parser.add_argument("--num_k", type=int, default=1)
parser.add_argument("--num_q", type=int, default=1)

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)
args.num_k = max(args.num_k, args.num_q)

# Load Tokenizer
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

# Define Needle/Haystack Format 
language = args.language
if language == "en":
    needle = "Document: {document}"
elif language == "zh":
    needle = "文档: {document}"
elif language == "es":
    needle = "Documento: {document}"
elif language == "fr":
    needle = "Document: {document}"
elif language == "de":
    needle = "Dokument: {document}"
elif language == "ru":
    needle = "Документ: {document}"
elif language == "ja":
    needle = "ドキュメント: {document}"
elif language == "th":
    needle = "เอกสาร: {document}"
elif language == "sw":
    needle = "Hati: {document}"
elif language == "bn":
    needle = "নথি: {document}"
elif language == "te":
    needle = "పత్రం: {document}"
elif language == "ar":
    needle = "الوثيقة: {document}"
elif language == "ko":
    needle = "문서: {document}"
elif language == "vi":
    needle = "Tài liệu: {document}"
elif language == "cs":
    needle = "Dokument: {document}"
elif language == "hu":
    needle = "Dokumentum: {document}"
elif language == "sr":
    needle = "Документ: {document}"
else:
    raise NotImplementedError(f'{args.language} is not supported.')

essay = f"/cpfs01/shared/XNLP_H800/huangxu/data/un_test/un_test.{args.language}.json"
essay = json.load(open(essay))['text']
haystack = essay.strip().split("\n")

def read_xquad(file):
    with open(file) as f:
        data = json.load(f)

    total_docs = [p['context'] for d in data['data'] for p in d['paragraphs'] if len(p["qas"]) > 1 or p["qas"][0]["answers"][0]["text"] != ""]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data['data']:
        for p in d['paragraphs']:
            if len(p["qas"]) == 1 and p["qas"][0]["answers"][0]["text"] == "":
                continue
            for qas in p['qas']:
                total_qas.append({
                    'query': qas['question'],
                    'outputs': [a['text'] for a in qas['answers']],
                    'context': [total_docs_dict[p['context']]],
                    'context_id': p['context_id']
                })

    return total_qas, total_docs

QAS, DOCS = read_xquad(f"/cpfs01/shared/XNLP_H800/huangxu/xeval/data/xquad/xquad_{language}.json")
# qas_indices = sorted(random.sample(range(len(QAS)), args.num_samples))

# Positions
DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))

def sample_qas(context_set):
    qas = random.choice(QAS)
    while qas["context_id"] in context_set:
        qas = random.choice(QAS)
    context_set.add(qas["context_id"])
    return qas


def generate_input_output(num_haystack):
    contexts, questions, ans = [], [], []
    context_id_set = set()
    for _ in range(args.num_k):
        qas = sample_qas(context_id_set)
        contexts.append(qas["context"][0])
        questions.append(qas["query"])
        ans.append(qas["outputs"][0])

    needles = [needle.format(document=DOCS[c]) for c in contexts]
    random.Random(args.random_seed).shuffle(needles)
    
    # Context
    document_sents = haystack[:num_haystack]
    insertion_positions = [0] + \
                            sorted([int(len(document_sents) * (depth / 100)) for depth in random.sample(DEPTHS, len(needles))]) + \
                            [len(document_sents)]
    document_sents_list = []
    for i in range(1, len(insertion_positions)):
        last_pos = insertion_positions[i - 1]
        next_pos = insertion_positions[i]
        document_sents_list.append(" ".join(document_sents[last_pos: next_pos]))
        if i-1 < len(needles):
            document_sents_list.append(needles[i - 1])
    input_context = " ".join(document_sents_list)


    ## Query and Answer
    indices = random.sample(range(args.num_k), args.num_q)
    queries = [questions[i] for i in indices]
    answers = [ans[i] for i in indices]
    query = '\n'.join(queries)
    
    template = args.template
    if args.num_q == 1:
        template = template.replace('questions', 'question')
        template = template.replace('Questions', 'Question')
        template = template.replace('answers', 'answer')
        template = template.replace('Answers', 'Answer')
        # template = template.replace('Each line has one answer to a question. ', '')

    input_text = template.format(
        context=input_context,
        query=query,
    )

    return input_text, answers


def generate_samples(num_samples: int, max_seq_length: int, save_dir: str, incremental: int = 500):
    write_jsons = []
    tokens_to_generate = args.tokens_to_generate

    incremental = 10

    num_haystack = incremental
        
    total_tokens = 0  # Track the total tokens generated for the first example
    while total_tokens + tokens_to_generate < max_seq_length :  
        input_text, answer = generate_input_output(num_haystack)
        # Calculate the number of tokens in the example
        total_tokens = len(TOKENIZER.text_to_tokens(input_text + ' '.join(answer)))
        print(f'Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Haystack: {num_haystack}')
        if total_tokens + tokens_to_generate > max_seq_length:
            num_haystack -= incremental
            break
    
        if num_haystack > len(haystack):
            num_haystack = len(haystack)
            break
        
        num_haystack += incremental

    print('Num haystack:', num_haystack)
    
    # Generate samples
    for index in tqdm(range(num_samples)):
        used_haystack = num_haystack
        while(True):
            try:
                input_text, answer  = generate_input_output(used_haystack)
                length = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:
                if used_haystack > incremental:
                    used_haystack -= incremental
        
        if args.remove_newline_tab:
            input_text = ' '.join(input_text.replace('\n', ' ').replace('\t', ' ').strip().split())

        formatted_output = {
            'index': index,
            "input": input_text,
            "outputs": answer,
            "length": length,
        }
        write_jsons.append(formatted_output)

    return write_jsons


def main():
    save_file = args.save_dir / f'{args.save_name}' / f'{args.subset}.jsonl'
    save_file.parent.mkdir(parents=True, exist_ok=True)

    write_jsons = generate_samples(
        num_samples=args.num_samples, 
        max_seq_length=args.max_seq_length,
        save_dir=args.save_dir
    )

    write_manifest(save_file, write_jsons, False)

if __name__ == "__main__":
    main()