from typing import Any, Callable, Dict, List
from dataclasses import dataclass
import json
import argparse
import re

import torch
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

from utils import build_functions
from output_parser import Llama3OutputParser



@dataclass
class Evaluator:
    model_name: str
    infer_backend: str
    output_parser_name: str

    # def __post_init__(self) -> None:
    #     self.toolllm_helper = ToolLLMEvaluationDataHelper(
    #         hf_path=self.hf_path,
    #         standardized_queries_subset=self.standardized_queries_subset,
    #     )

    def run(self) -> None:
        tools, functions = self.build_functions()
        self.build_toolllm_functions_queries()
        exit(0)
        self.build_model()
        self.build_parser()
        eval_dataset = self.get_eval_dataset()
        accuracies = []

        for i, sample in enumerate(eval_dataset):
            context_tools = [tools[k] for k in sample["context_functions"]]
            # context_functions = [functions[k] for k in sample["context_functions"]]
            prompt = sample["prompt"]
            message = [{"role": "user", "content": prompt}]
            error_message = None
            output = None
            original_output = None
            try:
                original_output = self.model.chat(message, sampling_params=self.sampling_params, tools=context_tools, use_tqdm=False)
                original_output = original_output[0].outputs[0].text.strip()
                output = self.output_parser.parse(original_output)
                if output[0] is not None and output[1] is not None:
                    output = functions[output[0]](**output[1])
            except Exception as e:  # pylint: disable=broad-exception-caught
                error_message = f"Original output\n\t{original_output}\nOutput\n\t{output}\nError message\n\t{str(e)}\n\n"
                output = (None, None)

            predicted_function_name, predicted_args_dict = output

            reference_function_name = sample["python_function_name"]
            reference_input_args_dict = json.loads(sample["python_args_dict"])
            _, reference_args_dict = functions[reference_function_name](
                **reference_input_args_dict
            )

            function_name_match = predicted_function_name == reference_function_name
            args_dict_match = predicted_args_dict == reference_args_dict
            accuracy = function_name_match and args_dict_match

            accuracy_str = "\033[32mCORRECT\033[0m" if accuracy else "\033[31mWRONG\033[0m"
            error_message_str = "" if error_message is None else error_message
            print(
                f"Example {i+1} / {len(eval_dataset)}\n\nPrompt\n\t{prompt}\n\nReference\n\t{reference_function_name}, {reference_args_dict}\n\nPrediction\n\t{predicted_function_name}, {predicted_args_dict}\n\n{accuracy_str}\n\n{error_message_str}{'-' * 80}\n"
            )

            accuracies.append(accuracy)

        accuracy = 100 * sum(accuracies) / len(eval_dataset)
        print(f"Accuracy: {accuracy:.3f}%")

    def get_eval_dataset(self) -> Dataset:
        d = load_dataset(
            path="Nexusflow/NexusRaven_API_evaluation",
            name="standardized_queries",
            split="train",
            cache_dir="/cpfs01/shared/XNLP_H800/huangxu/.cache/huggingface/datasets"
        )
        assert len(d) > 0
        # d = d.select(range(10))

        return d

    def build_model(self):
        match self.infer_backend:
            case "vllm":
                self.model = LLM(model=self.model_name, tensor_parallel_size=torch.cuda.device_count(), dtype="bfloat16", enforce_eager=True)
                self.sampling_params = SamplingParams(temperature=0.0, max_tokens=400)
            case _:
                raise NotImplementedError
        # if "openai" in self.model_name:
        #     api_key = environ.get("OPENAI_API_KEY", None)
        #     if api_key is None:
        #         api_key = input(
        #             f"Please input your OpenAI API key to use the `{self.model_name}` model: "
        #         )
        #         environ["OPENAI_API_KEY"] = api_key

        # gpt_params = {
        #     "temperature": 0,
        #     "verbose": True,
        # }
        # hf_params = {
        #     "max_new_tokens": 400,
        #     "do_sample": False,
        #     "inference_server_url": self.inference_server_url,
        #     "temperature": 0.001,
        # }

    def build_parser(self):
        match self.output_parser_name:
            case "llama3":
                self.output_parser = Llama3OutputParser()
            case _:
                raise NotImplementedError


    def build_functions(self):

        d = load_dataset(
            path="Nexusflow/NexusRaven_API_evaluation",
            name="standardized_api_list",
            split="train",
            cache_dir="/cpfs01/shared/XNLP_H800/huangxu/.cache/huggingface/datasets"
        )
        return build_functions(d)
    
    def build_toolllm_functions_queries(self):
        d = load_dataset(
            path="Nexusflow/NexusRaven_API_evaluation",
            name="raw_queries",
            split="train",
            cache_dir="/cpfs01/shared/XNLP_H800/huangxu/.cache/huggingface/datasets"
        )
        d = d.filter(lambda x: x == "toolllm", input_columns="dataset")

        functions = {}
        context_functions = []
        prompts = []
        reference_function_calls = []
        for example in d:
            example = json.loads(example["query_dict"])
            function_strs = example["context"]

            function_str = None
            function_name = None
            function_names = []
            for function_str in function_strs:
                # Replace braces since they cause problems for langchain's formatting
                function_str = function_str.replace("{", "")
                function_str = function_str.replace("}", "")

                function_name = function_str[: function_str.find("(")].removeprefix(
                    "def "
                )

                # Remove the function body to use our own for evaluation
                function_str = function_str[: function_str.find("args_dict")]
                description = function_str.split("\n", 1)[1].strip()
                description_regex = re.search(r"'(.*)'", description)
                function_str = f"""
{function_str}return ("{function_name}", {{k: int(v) if isinstance(v, float) else v for k, v in locals().items()}})
"""

                exec(function_str)

                function_names.append(function_name)

            namespace = locals()
            new_functions = {n: namespace[n] for n in function_names}
            assert all(n not in functions for n in new_functions.keys())
            functions.update(new_functions)

            context_functions.append(function_names)
            prompts.append(example["Input"])
            reference_function_calls.append(example["Output"])

        toolllm_dataset = Dataset.from_dict(
            {
                "context_functions": context_functions,
                "prompt": prompts,
                "reference_function_call": reference_function_calls,
            }
        )
        return toolllm_dataset, functions

    # def build_agent(self, llm: Any, tools: List[StructuredTool]) -> Any:
    #     match self.agent_name:
    #         case "OPENAI_FUNCTIONS":
    #             return initialize_agent(
    #                 tools,
    #                 llm,
    #                 agent=AgentType.OPENAI_FUNCTIONS,
    #                 verbose=True,
    #                 max_iterations=1,
    #             )
    #         case "SIMPLE":
    #             return initialize_agent(
    #                 tools,
    #                 llm,
    #                 agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    #                 verbose=True,
    #                 max_iterations=1,
    #             )
    #         case "SIMPLE_NONCHAT":
    #             return initialize_agent(
    #                 tools,
    #                 llm,
    #                 agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #                 verbose=True,
    #                 max_iterations=1,
    #             )
    #         case "NEXUSRAVEN":
    #             raven_prompt = RavenPromptTemplate(
    #                 template=RAVEN_PROMPT, tools=tools, input_variables=["input"]
    #             )
    #             llm_chain = LLMChain(llm=llm, prompt=raven_prompt)
    #             output_parser = RavenOutputParser()
    #             agent = LLMSingleActionAgent(
    #                 llm_chain=llm_chain,
    #                 output_parser=output_parser,
    #                 stop=["\nReflection:"],
    #                 allowed_tools=tools,
    #             )
    #             agent_chain = AgentExecutor.from_agent_and_tools(
    #                 agent=agent, tools=tools, verbose=True
    #             )
    #             return agent_chain
    #         case "TOOLLLM":
    #             return self._build_toolllm_agent(llm, tools)
    #         case "TOOLALPACA":
    #             return self._build_toolalpaca_agent(llm, tools)
    #         case _:
    #             raise KeyError(f"Invalid agent_name `{self.agent_name}`")

    # def _build_toolllm_agent(self, llm: Any, tools: List[StructuredTool]) -> Any:
    #     if self.agent is not None:
    #         return self.agent

    #     responses = load_dataset(
    #         self.hf_path,
    #         name=llm,
    #         split="train",
    #     )

    #     prompt_to_response = dict()
    #     for response in responses["response"]:
    #         prompt = response[0]["query"]
    #         response = response[1]["function_call"]
    #         prompt_to_response[prompt] = response

    #     agent = MagicMock()

    #     def run(prompt: str) -> str:
    #         return prompt_to_response.get(prompt, None)

    #     agent.run = run

    #     self.agent = agent
    #     return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--infer-backend", type=str, required=True)
    parser.add_argument("--output-parser-name", type=str, required=True)
    args = parser.parse_args()

    Evaluator(**vars(args)).run()
