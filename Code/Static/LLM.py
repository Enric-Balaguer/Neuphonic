from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

login(token="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")


def LLM_response(text_prompt:str):
    """
    Using Mistral 7B model, generate response from provided text_prompt.
    text_prompt: string prompt for Mistral 7B model.
    """
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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")
    cache_dir = 'Neuphonic/Models'
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto")
    
    # Tokenize the input
    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,  # Allowing for more extended output
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the generated tokens, ensuring to skip the tokens used for the prompt
    generated_tokens = outputs[:, inputs['input_ids'].size(1):][0]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return result.strip()