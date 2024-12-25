import json

import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
# from transformers import AutoTokenizer

import utils

# def vt_get_domain_report(domain: str, x_apikey: str):
#     """Retrieves a domain report. These reports contain information regarding the domain itself that VirusTotal has collected. Args: - domain: string, required, Domain name - x-apikey: string, required, Your API key"""
#     return ("vt_get_domain_report", locals())

# def vt_get_domain_report(domain: str, x_apikey: str):
#     """
#     Retrieves a domain report. These reports contain information regarding the domain itself that VirusTotal has collected.

#     Args:
#         domain: <PLACEHOLDER>
#         x_apikey: <PLACEHOLDER>
#     """
#     return ("vt_get_domain_report", locals())


def main():
    api_list = load_dataset("Nexusflow/NexusRaven_API_evaluation", "standardized_api_list", split="train", cache_dir="/cpfs01/shared/XNLP_H800/huangxu/.cache/huggingface/datasets")
    test_case = api_list.select([0, 1])
    tools, functions = utils.build_functions(test_case)

    model = LLM(model="/cpfs01/shared/XNLP_H800/hf_hub/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.95, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=400)
    # tokenizer = AutoTokenizer.from_pretrained("/cpfs01/shared/XNLP_H800/hf_hub/Llama-3.1-8B-Instruct")

    prompt = "Look for CPEs related to 'Windows 10'. I want to see the most recent 5 and print the request URL for debugging."
    messages = [{"role": "user", "content": prompt}]

    outputs = model.chat(messages, sampling_params=sampling_params, tools=list(tools.values()))
    print(outputs[0].outputs[0].text)




if __name__ == "__main__":    main()
