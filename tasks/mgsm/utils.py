import argparse

import yaml


LANGUAGES = {
    "bn": {  # Bengali
        # "QUESTION": "প্রশ্ন:",
        "QUESTION": "\u09aa\u09cd\u09b0\u09b6\u09cd\u09a8:",
        # "ANSWER": "ধাপে ধাপে উত্তর:",
        "ANSWER": "\u09a7\u09be\u09aa\u09c7 \u09a7\u09be\u09aa\u09c7 \u0989\u09a4\u09cd\u09a4\u09b0:",
        "DIRECT": "\u0989\u09a4\u09cd\u09a4\u09b0:",
        "REGEX": "The answer is (\\-?[0-9\\.\\,]+)",
    },
    "de": {  # German
        "QUESTION": "Frage:",
        # "ANSWER": "Schritt-für-Schritt-Antwort:",
        "ANSWER": "Schritt-f\u00fcr-Schritt-Antwort:",
        "DIRECT": "Antwort:",
        "REGEX": "Die Antwort lautet (\\-?[0-9\\.\\,]+)",
    },
    "en": {  # English
        "QUESTION": "Question:",
        "ANSWER": "Step-by-Step Answer:",
        "DIRECT": "Answer:",
        "REGEX": "The answer is (\\-?[0-9\\.\\,]+)",
    },
    "es": {  # Spanish
        "QUESTION": "Pregunta:",
        "ANSWER": "Respuesta paso a paso:",
        "DIRECT": "Respuesta:",
        "REGEX": "La respuesta es (\\-?[0-9\\.\\,]+)",
    },
    "fr": {  # French
        "QUESTION": "Question :",
        # "ANSWER": "Réponse étape par étape :"
        "ANSWER": "R\u00e9ponse \u00e9tape par \u00e9tape :",
        # "DIRECT": "Réponse :",
        "DIRECT": "R\u00e9ponse :",
        # "REGEX": "La réponse est (\\-?[0-9\\.\\,]+)",
        "REGEX": "La r\u00e9ponse est (\\-?[0-9\\.\\,]+)",
    },
    "ru": {  # Russian
        # "QUESTION": "Задача:",
        "QUESTION": "\u0417\u0430\u0434\u0430\u0447\u0430:",
        # "ANSWER": "Пошаговоерешение:",
        "ANSWER": "\u041f\u043e\u0448\u0430\u0433\u043e\u0432\u043e\u0435\u0440\u0435\u0448\u0435\u043d\u0438\u0435:",
        "DIRECT": "Отвечать:",
        # "REGEX": "Ответ — (\\-?[0-9\\.\\,]+)",
        "REGEX": "\u041e\u0442\u0432\u0435\u0442 \u2014 (\\-?[0-9\\.\\,]+)",
    },
    "sw": {  # Swahili
        "QUESTION": "Swali:",
        "ANSWER": "Jibu la Hatua kwa Hatua:",
        "DIRECT": "Jibu:",
        "REGEX": "Jibu ni (\\-?[0-9\\.\\,]+)",
    },
    "te": {  # Telugu
        # "QUESTION": "ప్రశ్న:",
        "QUESTION": "\u0c2a\u0c4d\u0c30\u0c36\u0c4d\u0c28:",
        # "ANSWER": "దశలవారీగా సమాధానం:",
        "ANSWER": "\u0c26\u0c36\u0c32\u0c35\u0c3e\u0c30\u0c40\u0c17\u0c3e \u0c38\u0c2e\u0c3e\u0c27\u0c3e\u0c28\u0c02:",
        "DIRECT": "సమాధానం:",
        # "REGEX": "సమాధానం (\\-?[0-9\\.\\,]+)",
        "REGEX": "\u0c38\u0c2e\u0c3e\u0c27\u0c3e\u0c28\u0c02 (\\-?[0-9\\.\\,]+)",
    },
    "th": {  # Thai
        # "QUESTION": "โจทย์:",
        "QUESTION": "\u0e42\u0e08\u0e17\u0e22\u0e4c:",
        # "ANSWER": "คำตอบทีละขั้นตอน:",
        "ANSWER": "\u0e04\u0e33\u0e15\u0e2d\u0e1a\u0e17\u0e35\u0e25\u0e30\u0e02\u0e31\u0e49\u0e19\u0e15\u0e2d\u0e19:",
        "DIRECT": "คำตอบ:",
        # "REGEX": "คำตอบคือ (\\-?[0-9\\.\\,]+)",
        "REGEX": "\u0e04\u0e33\u0e15\u0e2d\u0e1a\u0e04\u0e37\u0e2d (\\-?[0-9\\.\\,]+)",
    },
    "ja": {  # Japanese
        # "QUESTION": "問題:",
        "QUESTION": "\u554f\u984c:",
        # "ANSWER": "ステップごとの答え:",
        "ANSWER": "\u30b9\u30c6\u30c3\u30d7\u3054\u3068\u306e\u7b54\u3048:",
        "DIRECT": "答え：",
        # "REGEX": "答えは(\\-?[0-9\\.\\,]+)です。",
        "REGEX": "\u7b54\u3048\u306f(\\-?[0-9\\.\\,]+)\u3067\u3059\u3002",
    },
    "zh": {  # Chinese
        # "QUESTION": "问题:",
        "QUESTION": "\u95ee\u9898:",
        # "ANSWER": "逐步解答:",
        "ANSWER": "\u9010\u6b65\u89e3\u7b54:",
        "DIRECT": "答案:",
        # "REGEX": "答案是 (\\-?[0-9\\.\\,]+)。",
        "REGEX": "\u7b54\u6848\u662f (\\-?[0-9\\.\\,]+)\u3002",
    },
    "ar": {  # Arabic
        # "QUESTION": "السؤال:",
        "QUESTION": "\u0627\u0644\u0633\u0624\u0627\u0644:",
        # "ANSWER": "الإجابة خطوة بخطوة:",
        "ANSWER": "\u0627\u0644\u0625\u062c\u0627\u0628\u0629 \u062e\u0637\u0648\u0629 \u0628\u062e\u0637\u0648\u0629:",
        "DIRECT": "إجابة:",
        # "REGEX": "الإجابة هي (\\-?[0-9\\.\\,]+)",
        "REGEX": "\u0627\u0644\u0625\u062c\u0627\u0628\u0629 \u0647\u064a (\\-?[0-9\\.\\,]+)",
    },
    "ko": {  # Korean
        # "QUESTION": "질문:",
        "QUESTION": "\uc9c8\ubb38:",
        # "ANSWER": "단계별 답변:",
        "ANSWER": "\ub2e8\uacc4\ubcc4 \ub2f5\ubcc0:",
        "DIRECT": "답변:",
        # "REGEX": "답은 (\\-?[0-9\\.\\,]+)입니다\\.",
        "REGEX": "\ub2f5\uc740 (\\-?[0-9\\.\\,]+)\uc785\ub2c8\ub2e4\\.",
    },
    "vi": {  # Vietnamese
        # "QUESTION": "Câu hỏi:",
        "QUESTION": "C\xe2u h\u1ecfi:",
        # "ANSWER": "Câu trả lời từng bước:",
        "ANSWER": "C\xe2u tr\u1ea3 l\u1eddi t\u1eebng b\u01b0\u1edbc:",
        "DIRECT": "Trả lời:",
        # "REGEX": "Câu trả lời là (\\-?[0-9\\.\\,]+)",
        "REGEX": "C\xe2u tr\u1ea3 l\u1eddi l\xe0 (\\-?[0-9\\.\\,]+)",
    },
    "cs": {  # Czech
        # "QUESTION": "Otázka:",
        "QUESTION": "Ot\xe1zka:",
        # "ANSWER": "Odpověď krok za krokem:",
        "ANSWER": "Odpov\u011b\u010f krok za krokem:",
        "DIRECT": "Odpověď:",
        # "REGEX": "Odpověď je (\\-?[0-9\\.\\,]+)",
        "REGEX": "Odpov\u011b\u010f je (\\-?[0-9\\.\\,]+)",
    },
    "hu": {  # Hungarian
        # "QUESTION": "Kérdés:",
        "QUESTION": "K\xe9rd\xe9s:",
        # "ANSWER": "Válasz lépésről lépésre:",
        "ANSWER": "V\xe1lasz l\xe9p\xe9sr\u0151l l\xe9p\xe9sre:",
        "DIRECT": "Válasz:",
        # "REGEX": "A válasz (\\-?[0-9\\.\\,]+)",
        "REGEX": "A v\xe1lasz (\\-?[0-9\\.\\,]+)",
    },
    "sr": {  # Serbian
        # "QUESTION": "Питање:",
        "QUESTION": "\u041f\u0438\u0442\u0430\u045a\u0435:",
        # "ANSWER": "Одговор корак по корак:",
        "ANSWER": "\u041e\u0434\u0433\u043e\u0432\u043e\u0440 \u043a\u043e\u0440\u0430\u043a \u043f\u043e \u043a\u043e\u0440\u0430\u043a:",
        "DIRECT": "одговор:",
        # "REGEX": "Одговор је (\\-?[0-9\\.\\,]+)",
        "REGEX": "\u041e\u0434\u0433\u043e\u0432\u043e\u0440 \u0458\u0435 (\\-?[0-9\\.\\,]+)",
    },
}


def add_regex_pattern(regex_pattern):
    if regex_pattern is None:
        return {}
    return {
        "filter_list": [
            {
                "name": "strict-match",
                "filter": [
                    {
                        "function": "regex",
                        "regex_pattern": f"""{regex_pattern}""",
                    },
                    {
                        "function": "take_first",
                    },
                ],
            },
            {
                "name": "flexible-extract",
                "filter": [
                    {
                        "function": "regex",
                        "regex_pattern": """(-?[$0-9.,]{2,})|(-?[0-9]+)""",
                        "group_select": -1,
                    },
                    {
                        "function": "take_first",
                    },
                ],
            },
        ],
    }


def gen_lang_yamls(output_dir: str, overwrite: bool, mode: str) -> None:
    """
    Generate a yaml file for each language.

    :param output_dir: The directory to output the files to.
    :param overwrite: Whether to overwrite files if they already exist.
    """
    err = []
    for lang in LANGUAGES.keys():
        try:
            QUESTION = LANGUAGES[lang]["QUESTION"]

            yaml_template = "cot_yaml"
            filter_list = {}
            DELIMITER = None
            if mode == "direct":
                ANSWER = LANGUAGES[lang]["DIRECT"]
                REGEX = None
                task_name = f"xmgsm_direct_{lang}"
                yaml_template = "direct_yaml"
            elif mode == "native-cot":
                ANSWER = LANGUAGES[lang]["ANSWER"]
                REGEX = LANGUAGES[lang]["REGEX"]
                task_name = f"xmgsm_native_cot_{lang}"
                filter_list = add_regex_pattern(REGEX)
                DELIMITER = "" if lang in ["zh", "ja"] else None
            elif mode == "en-cot":
                ANSWER = LANGUAGES["en"]["ANSWER"]
                REGEX = LANGUAGES["en"]["REGEX"]
                task_name = f"xmgsm_en_cot_{lang}"

            file_name = f"{task_name}.yaml"
            ANSWER_TO_SKIP = len(LANGUAGES[lang]["ANSWER"]) + 1
            with open(
                f"{output_dir}/{file_name}", "w" if overwrite else "x", encoding="utf8"
            ) as f:
                f.write("# Generated by utils.py\n")
                yaml.dump(
                    {
                        "include": yaml_template,
                        "dataset_name": lang,
                        "task": f"{task_name}",
                        "doc_to_text": f"""{{% if answer is not none %}}"""
                        f"""{{{{question+"\\n{ANSWER}"}}}}"""
                        f"""{{% else %}}"""
                        f"""{{{{"{QUESTION} "+question+"\\n{ANSWER}"}}}}"""
                        f"""{{% endif %}}""",
                        "doc_to_target": f"""{{% if answer is not none %}}"""
                        f"""{{{{answer[{ANSWER_TO_SKIP}:]}}}}"""
                        f"""{{% else %}}"""
                        f"""{{{{answer_number|string}}}}"""
                        f"""{{% endif %}}""",
                        **filter_list,
                        "generation_kwargs": {
                            "until": ["</s>", "<|im_end|>"],
                            "do_sample": False,
                            "max_gen_toks": 1024,
                        },
                        **({"target_delimiter": DELIMITER} if DELIMITER else {}),
                    },
                    f,
                    allow_unicode=True,
                    width=float("inf"),
                )
        except FileExistsError:
            err.append(file_name)

    if len(err) > 0:
        raise FileExistsError(
            "Files were not created because they already exist (use --overwrite flag):"
            f" {', '.join(err)}"
        )


def main() -> None:
    """Parse CLI args and generate language-specific yaml files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite files if they already exist",
    )
    parser.add_argument(
        "--output-dir", default=".", help="Directory to write yaml files to"
    )
    parser.add_argument(
        "--mode",
        default="native-cot",
        choices=["direct", "native-cot", "en-cot"],
        help="Mode of chain-of-thought",
    )
    args = parser.parse_args()

    gen_lang_yamls(output_dir=args.output_dir, overwrite=args.overwrite, mode=args.mode)


if __name__ == "__main__":
    main()
