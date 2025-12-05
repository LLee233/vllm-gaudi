import os

from vllm import LLM, SamplingParams

os.environ["VLLM_SKIP_WARMUP"] = "true"
# prompts = [
#     "Hello, my name is",
#     "Tell me the answer of 1+1 (Just give me the answer and stop.).",
#     "0.999 compares to 0.9 is ",
#     "The capital of France is",
#     "The future of AI is",
#     "Tell me a joke about computers.",
# ]

prompts = [
    "A store sells notebooks for $2.50 each. If Emma buys 7 notebooks and uses a 20% off coupon on the total, how much does she pay?",
    "A rectangular garden is 12 meters long and 9 meters wide. What is its perimeter?",
    "A train travels at 60 miles per hour. How long does it take to go 270 miles?",
    "There are 28 students in a class. The number of girls is 6 more than the number of boys. How many boys are in the class?",
    "A tank contains 150 liters of water. 35 liters are added, and then 48 liters are removed. How many liters are left?",
    "A baker made 120 cookies. He packs them into boxes of 8 cookies each and sells 11 boxes. How many cookies remain unpacked?",
    "A cyclist rides 18 miles in the morning and 24 miles in the afternoon. If his average speed for the whole day is 12 miles per hour, how many hours did he ride?",
    "A book has 320 pages. Olivia reads 40 pages per day for 6 days. How many pages are left to read?",
    "A shop offers a “buy 3, get 1 free” deal on pens. Each pen costs $1.80. If Noah wants 12 pens, how much will he pay?",
    "A rectangle has area 84 square units and length 14 units. What is its width?",
]

sampling_params = SamplingParams(temperature=0.1, max_tokens=300, stop=["\nHuman:", "<eos>", "Q:", "END token string", "</s>"])
                                                                        # , "."])
# model = "/mnt/weka/data/huggingface-models/vLLM/text_models/generative_modes/text_generation/Qwen--Qwen3-30B-A3B"
# model = "/mnt/weka/data/huggingface-models/vLLM/text_models/generative_modes/text_generation/Qwen--Qwen3-8B"
# model = "/mnt/weka/data/pytorch/Qwen/Qwen3-32B/"
model = "/software/users/xinyili1/Qwen3-Next-80B-A3B-Instruct/"
# model = "/home/xinyili1/DeepSeek-OCR_hf/"

# model = "/mnt/weka/llm/Qwen3/Qwen3-30B-A3B/"
# model = "/mnt/weka/data/huggingface-models/vLLM/text_models/generative_modes/text_generation/Qwen--Qwen3-30B-A3B"
# model = "/mnt/weka/data/huggingface-models/vLLM/text_models/generative_modes/text_generation/Qwen--Qwen3-8B"
# model = "/mnt/weka/llm/Qwen3/Qwen3-32B/"
# model = "/mnt/weka/data/pytorch/Qwen/Qwen3-32B/"
# model = "meta-llama/Llama-3.2-1B-Instruct"
# model = "/mnt/weka/llm/DeepSeek-V2-Lite-Chat/"
# model = "/mnt/weka/data/mlperf_models/Mixtral-8x7B-Instruct-v0.1"
# model = "/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B/"

# kwargs = {"tensor_parallel_size": 1}
kwargs = {"tensor_parallel_size": 4}
if os.path.basename(model) in ["Qwen3-8B","Qwen3-30B-A3B", "DeepSeek-V2-Lite-Chat", "DeepSeek-OCR"]:
    kwargs["enable_expert_parallel"] = True
llm = LLM(model=model, max_model_len=4096, trust_remote_code=True, enforce_eager=True, gpu_memory_utilization=0.5, **kwargs)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
