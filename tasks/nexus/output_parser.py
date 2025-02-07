import json
import re

class Llama3OutputParser:
    def __init__(self) -> None:
        self.pattern = re.compile(r"<\|python_tag\|>(.*)")

    def parse(self, text: str):
        m = re.match(self.pattern, text)
        if m is None:
            return None, None
        text = m.group(1)
        output = json.loads(text)
        return output["name"], output["parameters"]

class GenericOutputParser:
    def __init__(self) -> None:
        self.patterns = [
            re.compile(r"```json([\s\S]*?)```"),
            re.compile(r"```([\s\S]*?)```"),
            re.compile(r"({[\s\S]*})")
        ]

    def parse(self, text: str):
        for pattern in self.patterns:
            m = re.search(pattern, text)
            if m is not None:
                try:
                    text = m.group(1)
                    output = json.loads(text)
                    name = output["name"]
                    parameters = output["parameters"]
                    return name, parameters
                except (AttributeError, json.JSONDecodeError, KeyError):
                    continue
        raise ValueError("No valid output found")

class Qwen2OutputParser:
    def __init__(self) -> None:
        self.pattern = re.compile(r"<tool_call>\n(.*)?\n</tool_call>")

    def parse(self, text: str):
        m = re.match(self.pattern, text)
        if m is None:
            return None, None
        text = m.group(1)
        output = json.loads(text)
        return output["name"], output["arguments"]