import os
import sys
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append("/nas/shared/NLP_A100/huangxu/xeval/tasks/xeval/ifeval")
import dataclasses
from typing import Dict, Optional, Union


import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]
    lang: str


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(
    inp,
    response,
):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id, inp.lang)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    response,
):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id, inp.lang)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
        lang=doc.get("lang", "en")
    )
    response = results[0]

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc

if __name__ == "__main__":
    doc = {"key": 1609, "prompt": "有什么好办法可以约 Sonia 出去？请用 4 段文字回复，每段用两行隔开。请用双引号将整个回复括起来。第一段必须以单词“周末”开头。", "instruction_id_list": ["length_constraints:nth_paragraph_first_word", "startend:quotation"], "kwargs": [{"num_highlights": None, "relation": None, "num_words": None, "num_placeholders": None, "prompt_to_repeat": None, "num_bullets": None, "keywords": None, "num_paragraphs": 4, "language": None, "section_spliter": None, "num_sections": None, "end_phrase": None, "forbidden_words": None, "keyword": None, "frequency": None, "postscript_marker": None, "num_sentences": None, "first_word": "周末", "nth_paragraph": 1}, {"num_highlights": None, "relation": None, "num_words": None, "num_placeholders": None, "prompt_to_repeat": None, "num_bullets": None, "keywords": None, "num_paragraphs": None, "language": None, "section_spliter": None, "num_sections": None, "end_phrase": None, "forbidden_words": None, "keyword": None, "frequency": None, "postscript_marker": None, "num_sentences": None, "first_word": None, "nth_paragraph": None}], "lang": "zh-CN"}
    resp = "\"周末的好时光，适合出去玩耍。可以邀请Sonia去公园或是游乐场，享受美好的时光。\"\n\n\"或者可以邀请她去看电影或是听音乐会，享受艺术的魅力。\"\n\n\"如果Sonia喜欢运动，可以邀请她去打球或是去健身房，锻炼身体。\"\n\n\"最后，可以邀请她去吃饭或是喝咖啡，享受美食和温馨的时光。\""
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
        lang=doc.get("lang", "en")
    )
    out_loose = test_instruction_following_loose(inp, resp)
    print(out_loose)
