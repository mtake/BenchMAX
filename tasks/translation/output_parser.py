from typing import List

def build_output_parser(name: str=None):
    match name:
        case None:
            return BaseOutputParser()
        case "r1_distill":
            return R1DistillModelOutputParser()
        case _:
            raise NotImplementedError

class BaseOutputParser():
    def parse(self, texts: List[str]):
        return texts

class R1DistillModelOutputParser(BaseOutputParser):
    def parse(self, texts: List[str]):
        return [text.split("</think>")[-1].strip() for text in texts]