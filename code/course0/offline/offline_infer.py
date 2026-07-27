import os

os.environ["VLLM_USE_V1"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams

def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "Write a poem about China:",
        "Who won the world series in 2020?",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="/root/model/Qwen3-0.6B", enforce_eager=True)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()
