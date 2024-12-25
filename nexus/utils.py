from typing import Any, Callable, Dict, List, Tuple, Union
import re
import json
import ast

from datasets import Dataset
from transformers import utils


def build_functions(d: Dataset) -> Dict[str, Callable]:
    function_names = []
    tools = {}
    for function_dict in d:
        name = function_dict["name"]
        description = function_dict["description"].strip("\n")
        if description.startswith("    "):
            description = description.strip().replace("\n    ", "\n")
        args_dicts = function_dict["args_dicts"]

        args_dicts = sorted(args_dicts, key=lambda d: d["required"], reverse=True)

        args_docstring = "\n\n    Args:" if args_dicts else ""
        args_strs = []
        for arg_dict in args_dicts:
            if arg_dict["required"]:
                default_str = ""
            else:
                default = arg_dict["default"]
                if isinstance(default, str) and default.startswith("JSON:"):
                    default = default.removeprefix("JSON:")
                    default = json.loads(default)
                default_str = f" = {default}"

            if arg_dict["type"] == "None":
                type_str_m = re.search(arg_dict["name"] + r" \((.+?)\)", description)
                assert type_str_m is not None
                type_str = type_str_m.group(1)
                if "datetime" in type_str:
                    type_str = ": Union[str, object]"
                else:
                    type_str = f": {type_str}"
                # type_str = ""
            else:
                type_str = f": {arg_dict['type']}"

            args_str = f"{arg_dict['name']}{type_str}{default_str}"
            args_strs.append(args_str)
            args_docstring += f"\n        {arg_dict['name']}: <arg description>"

        function_str = f'def {name}({", ".join(args_strs)}):\n    """\n    <description>{args_docstring}\n    """\n    return ("{name}", locals())'
        if locals().get(name, None) is not None:
            raise RuntimeError
        exec(function_str)
        function_names.append(name)
        tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
            }
        }

    functions = {}
    namespace = locals()
    for f in function_names:
        functions[f] = namespace[f]
        parameters = utils.get_json_schema(namespace[f])["function"]["parameters"]
        for v in parameters["properties"].values():
            v["description"] = ""
        tools[f]["function"]["parameters"] = parameters
    return tools, functions

# def get_json_schema(func: Callable):
#     tool = utils.get_json_schema(func)
#     for params in tool["function"]["parameters"]["properties"]:
#         for attr in params.values():
#             attr["description"] = ""
#     return tool


def parse_function_call_to_name_and_args(
    function: str,
) -> Tuple[str, List[str], Dict[str, Any]]:
    function = ast.parse(function).body[0]
    function_name = function.value.func.id
    keywords = function.value.keywords
    keywords = {k.arg: ast.literal_eval(k.value) for k in keywords}

    args = [ast.literal_eval(arg) for arg in function.value.args]

    """
    We use integers only for evaluation data since floats are tricky to evaluate
    e.g. the string representation of "3.33" may be "3.329999999" which yields an exact match of 0
    """
    keywords = {k: int(v) if isinstance(v, float) else v for k, v in keywords.items()}

    return function_name, args, keywords
