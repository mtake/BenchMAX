import torch
from vllm import LLM, SamplingParams

def main():
    model_path = "/cpfs01/shared/XNLP_H800/hf_hub/Llama-3.1-8B-Instruct"
    model = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.95, enforce_eager=True, dtype="bfloat16")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    messages = [[{"role": "user", "content": d}] for d in []]
    outputs = model.chat(messages, sampling_params=sampling_params)
    outputs = [o.outputs[0].text for o in outputs]

if __name__ == "__main__":
    main()