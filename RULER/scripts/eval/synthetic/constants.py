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
# limitations under the License.

"""
Add a new task:

TASK_NAME: {
    'metric_fn': the metric function with input (predictions: [str], references: [[str]]) to compute score.
}
"""

import sys
import re
import string
import unicodedata
from collections import Counter


PUNCT = {chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')}.union(string.punctuation)
WHITESPACE_LANGS = ['en', 'es', 'vi', 'de', 'ar', 'fr', 'ru', 'th', 'sw', 'bn', "te", "ko", "cs", "hu", "sr"]
MIXED_SEGMENTATION_LANGS = ['zh', 'ja']

def string_match_part(preds, refs, lang):
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

def string_match_all(preds, refs, lang):
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)


def whitespace_tokenize(text):
    return text.split()

def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', char) or char in PUNCT:
            if temp_str != "":
                ss = whitespace_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def normalize_answer(s, lang):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text, lang):
        if lang == 'en':
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        elif lang == 'es':
            return re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
        elif lang == 'vi':
            return re.sub(r'\b(của|là|cái|chiếc|những)\b', ' ', text)
        elif lang == 'de':
            return re.sub(r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b', ' ', text)
        elif lang == 'ar':
            return re.sub('\sال^|ال', ' ', text)
        elif lang == 'fr':
            return re.sub(r'\b(le|la|l\'|les|un|une|des)\b', ' ', text)
        elif lang == 'hu':
            return re.sub(r'\b(a|az|egy)\b', ' ', text)
        elif lang in ['zh', 'ru', 'ja', 'th', 'sw', 'bn', 'te', 'ko', 'cs', 'sr']:
            return text # These languages do not have formal articles
        else:
            raise Exception('Unknown Language {}'.format(lang))

    def white_space_fix(text, lang):
        if lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            raise Exception('Unknown Language {}'.format(lang))
        return ' '.join([t for t in tokens if t.strip() != ''])

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)

def f1_score(prediction, ground_truth, lang):
    prediction_tokens = normalize_answer(prediction, lang).split()
    ground_truth_tokens = normalize_answer(ground_truth, lang).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def word_f1(preds, refs, lang):
    preds = [pred.strip() for pred in preds]
    # assert all(len(pred) == len(refs[0]) for pred in preds)
    refs = ["\n".join(r) for r in refs]
    score = sum(f1_score(pred, ref, lang) for pred, ref in zip(preds, refs)) / len(refs) * 100
    return round(score, 2)

def match_and_f1(preds, refs, lang):
    return (string_match_all(preds, refs, lang), word_f1(preds, refs, lang))


TASKS = {
    'niah': {
        'metric_fn': string_match_all,
    },
    'qaiah': {
        'metric_fn': match_and_f1,
    },
    'variable_tracking': {
        'metric_fn': string_match_all,
    },
    'common_words_extraction': {
        'metric_fn': string_match_all,
    },
    'freq_words_extraction': {
        'metric_fn': string_match_all
    },
    'qa': {
        'metric_fn': string_match_part,
    },
}
