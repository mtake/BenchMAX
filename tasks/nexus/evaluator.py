from dataclasses import dataclass
import json
import argparse
import os

from datasets import Dataset, load_dataset
from vllm import SamplingParams

from utils import build_functions
from output_parser import Llama3OutputParser, GenericOutputParser, Qwen2OutputParser, R1DistillModelOutputParser
from models import VLLMCausalLLM, OpenAILLM



@dataclass
class Evaluator:
    model_name: str
    infer_backend: str
    output_parser_name: str
    testset_name: str
    output_dir: str = "./output"
    tensor_parallel_size: int = None
    trust_remote_code: bool = False
    tokenizer: str = None
    tokenizer_support_tools: bool = None
    max_tokens: int = 512
    temperature: float = 0.0

    base_url: str = None

    def run(self) -> None:
        tools, functions = self.build_functions()
        eval_dataset = self.get_eval_dataset()
        self.build_model()
        self.build_parser()
        accuracies = []

        original_outputs = []
        for i, sample in enumerate(eval_dataset):
            context_tools = [tools[k] for k in sample["context_functions"]]
            prompt = sample["prompt"]
            message = [{"role": "user", "content": prompt}]
            error_message = None
            output = None
            original_output = None
            original_output = self.model.chat(message, sampling_params=self.sampling_params, tools=context_tools)
            original_outputs.append({"text": original_output})
            try:
                output = self.output_parser.parse(original_output)
                if output[0] is not None and output[1] is not None:
                    output = functions[output[0]](**output[1])
            except Exception as e:
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
            original_outputs[-1]["accuracy"] = int(accuracy)

            accuracy_str = "\033[32mCORRECT\033[0m" if accuracy else "\033[31mWRONG\033[0m"
            error_message_str = "" if error_message is None else error_message
            print(
                f"Example {i+1} / {len(eval_dataset)}\n\nPrompt\n\t{prompt}\n\nReference\n\t{reference_function_name}, {reference_args_dict}\n\nPrediction\n\t{predicted_function_name}, {predicted_args_dict}\n\n{accuracy_str}\n\n{error_message_str}{'-' * 80}\n"
            )

            accuracies.append(accuracy)

        accuracy = 100 * sum(accuracies) / len(eval_dataset)
        print(f"Accuracy: {accuracy:.3f}%")

        testset_name = "nexus_" + self.testset_name
        model_name = os.path.basename(self.model_name)
        os.makedirs(os.path.join(self.output_dir, model_name), exist_ok=True)
        output_path = f"{self.output_dir}/{model_name}/{model_name}_{testset_name}_{self.output_parser_name}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "accuracy": accuracy,
                    "outputs": original_outputs,
                },
                f,
                indent=4,
            )

    def get_eval_dataset(self) -> Dataset:
        d = load_dataset(
            path="LLaMAX/BenchMAX_Multiple_Functions",
            name=self.testset_name,
            split="train",
        )
        return d

    def build_model(self):
        match self.infer_backend:
            case "vllm":
                self.model = VLLMCausalLLM(
                    model_path=self.model_name,
                    tokenizer=self.tokenizer,
                    tensor_parallel_size=self.tensor_parallel_size,
                    trust_remote_code=self.trust_remote_code,
                    tokenizer_support_tools=self.tokenizer_support_tools,
                )
                self.sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens)
            case "openai_api":
                api_key = os.getenv("OPENAI_API_KEY", None)
                if api_key is None:
                    raise ValueError(
                        "API key not found. Please set the `OPENAI_API_KEY` environment variable."
                    )
                self.model = OpenAILLM(model_name=self.model_name, base_url=self.base_url, api_key=api_key)
                self.sampling_params = {
                    "temperature": self.temperature,
                    "max_completion_tokens": self.max_tokens,
                }
            case _:
                raise NotImplementedError

    def build_parser(self):
        match self.output_parser_name:
            case "llama3":
                self.output_parser = Llama3OutputParser()
            case "generic":
                self.output_parser = GenericOutputParser()
            case "qwen2":
                self.output_parser = Qwen2OutputParser()
            case "r1_distill":
                self.output_parser = R1DistillModelOutputParser()
            case _:
                raise NotImplementedError


    def build_functions(self):
        d = load_dataset(
            path="Nexusflow/NexusRaven_API_evaluation",
            name="standardized_api_list",
            split="train",
        )
        return build_functions(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-m", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tokenizer-support-tools", type=lambda x: (str(x).lower() == "true") if x is not None else None, default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--infer-backend", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--testset-name", "-t", type=str, required=True)
    parser.add_argument("--output-parser-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output")
    
    args = parser.parse_args()

    Evaluator(**vars(args)).run()
