from typing import Callable, Dict
import re
import json

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

def get_tool_prompt(tools):
    tool_strings = [json.dumps(tool, indent=4, ensure_ascii=False) for tool in tools]
    tool_prompt = "Respond to the human as helpfully and accurately as possible. You have access to the following tools:\n\n"
    tool_prompt += "\n\n".join(tool_strings)
    tool_prompt += """\n\nReminder to ALWAYS respond with a valid json. Use tools if necessary. Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables or output comments."""
    return tool_prompt

