import json
import re

class Llama3OutputParser:
    def __init__(self) -> None:
        self.pattern = re.compile(r"<\|python_tag\|>(.*)")

    def parse(self, text: str):
        m = re.match(self.pattern, text)
        if m is None:
            return None, None,
        text = m.group(1)
        output = json.loads(text)
        return output["name"], output["parameters"]