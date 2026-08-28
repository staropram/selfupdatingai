import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

def test_gwen3():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    messages = [
        {
            "role": "user",
            "content": "In one sentence, explain why the sky looks blue.",
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0, input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    assert(response=="The sky looks blue because molecules in the atmosphere scatter blue light more than other colors due to its shorter wavelength, making blue light more visible during the day.")

    print(response)

if __name__=="__main__":
    test_gwen3()