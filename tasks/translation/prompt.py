
LANG_DICT = {
    "en": "English", "zh": "Simplified Chinese", "hu": "Hungarian", "es": "Spanish", "fr": "French", "de": "German", "ru": "Russian", "ja": "Japanese", "th": "Thai", "sw": "Swahili", "bn": "Bengali", "te": "Telugu", "ar": "Arabic", "ko": "Korean", "vi": "Vietnamese", "cs": "Czech", "sr": "Cyrillic Serbian"
}

PROMPTS = {
    "general": "Translate the following text from {src_lang} to {tgt_lang}.\n\n{src_lang} source:\n{src}\n\n{tgt_lang} translation:",
    "flores": "Translate the following text from {src_lang} to {tgt_lang}.\n\n{src_lang} source:\n{src}\n\n{tgt_lang} translation:",
    "ted": "Translate the following text from {src_lang} to {tgt_lang}.\n\n{src_lang} source:\n{src}\n\n{tgt_lang} translation:",
    "wmt24": "Translate the following text from {src_lang} to {tgt_lang}.\n\n{src_lang} source:\n{src}\n\n{tgt_lang} translation:",
    "ifeval": "Translate the following text from {src_lang} to {tgt_lang}. Only output the translation without any additional text.\n\n{src_lang} source:\n\"\"\"\n{src}\n\"\"\"\n\n{tgt_lang} translation:",
    "arenahard": "Translate the following text from {src_lang} to {tgt_lang}. Only output the translation without any additional text.\n\n{src_lang} source:\n\"\"\"\n{src}\n\"\"\"\n\n{tgt_lang} translation:",
    "mgsm": "Translate the following text from {src_lang} to {tgt_lang}. Only output the translation without any additional text.\n\n{src_lang} source:\n{src}\n\n{tgt_lang} translation:",
    "gpqa": "Translate the following text from {src_lang} to {tgt_lang}, ensuring that the option markers remain unchanged. Only output the translation without any additional text.\n\n{src_lang} source:\n{src}\n\n{tgt_lang} translation:",
    "lcb_v4": "Please translate the following {src_lang} coding problems into {tgt_lang}, adhering to these specific guidelines:\n1. Do not translate any LaTeX code.\n2. Do not translate content representing code input/output or programming language syntax.\n3. Maintain the original formatting of the text and structure.\n4. Only output the translation without any additional comments or explanations.\n\n{src_lang} source:\n{src}\n\n{tgt_lang} translation:",
    "humaneval": "Please translate the following {src_lang} Python codes into {tgt_lang}, adhering to these specific guidelines:\n1. Do not translate content representing code input/output or programming language syntax. Only translate content in comments.\n2. Maintain the original formatting of the text, structure and indentation.\n3. Do not translate any LaTeX code.\n4. Only output the translation without any additional comments or explanations.\n\n{src_lang} source:\n{src}\n\n{tgt_lang} translation:",
    "nexus": "Translate the following text from {src_lang} to {tgt_lang} while keeping the specified keywords unchanged. Only output the translation without any additional text.\n\nKeywords:{args}\n\n{src_lang} source:\n{src}\n\n{tgt_lang} translation:",
}
