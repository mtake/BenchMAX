import json
import os

import torch
from vllm import LLM
import openai

from utils import get_tool_prompt

class VLLMCausalLLM:
    def __init__(self, model_path: str, tokenizer: str = None, tensor_parallel_size: int = 1, trust_remote_code: bool = False, tokenizer_support_tools: bool = None, **kwargs):
        if tensor_parallel_size is None or tensor_parallel_size < 0:
            tensor_parallel_size = torch.cuda.device_count()
        self.model = LLM(
            model=model_path,
            tokenizer=tokenizer,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            enforce_eager=True,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        if tokenizer_support_tools is None:
            tokenizer = self.model.get_tokenizer()
            self.tokenizer_support_tools = "tool" in tokenizer.chat_template
        else:
            self.tokenizer_support_tools = tokenizer_support_tools
        print(f"{self.tokenizer_support_tools=}")

    def chat(self, message, sampling_params=None, tools=None):
        if self.tokenizer_support_tools:
            response = self.model.chat(message, sampling_params=sampling_params, tools=tools, use_tqdm=False)
            output = response[0].outputs[0].text.strip()
            return output
        else:
            tool_prompt = get_tool_prompt(tools)
            if message[0]["role"] == "user":
                message[0]["content"] = tool_prompt + "\n\n" + message[0]["content"]
            elif message[0]["role"] == "system":
                message[0]["content"] += "\n\n" + tool_prompt
            response = self.model.chat(message, sampling_params=sampling_params, tools=tools, use_tqdm=False)
            output = response[0].outputs[0].text.strip()
            return output

class OpenAILLM:
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None):
        if api_key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def chat(self, message, sampling_params=None, tools=None):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message,
                max_completion_tokens=sampling_params["max_completion_tokens"],
                temperature=sampling_params["temperature"],
                tools=tools,
            )
        except openai.BadRequestError as e:
            if e.body["type"] == "invalid_request_error" and e.body["code"] == "string_above_max_length":
                tool_prompt = get_tool_prompt(tools)
                message.insert(0, {"role": "system", "content": tool_prompt})
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    max_completion_tokens=sampling_params["max_completion_tokens"],
                    temperature=sampling_params["temperature"]
                )
            else:
                raise e

        if response.choices[0].message.tool_calls is None:
            output = response.choices[0].message.content.strip()
        else:
            function = response.choices[0].message.tool_calls[0].function
            parameters = json.loads(function.arguments)
            output = json.dumps({"name": function.name, "parameters": parameters})
        return output