from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import re

def clean_response(response):
    # Regex to find and remove tags like [CISCO]...[/CISCO]
    cleaned_response = re.sub(r'\[.*?\].*?\[/.*?\]', '', response)
    return cleaned_response.strip()

login(token="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")

# Initialize model and tokenizer outside the function to avoid reloading
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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")
cache_dir = 'Neuphonic/Models'
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto")

def LLM_response(text_prompt: str):
    """
    Using Mistral 7B model, generate a more complete response from provided text_prompt.
    """
    # Prepare the prompt with context or instructions
    full_prompt = f"<s>[INST] {text_prompt} [/INST]"

    # Tokenize the input
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Generate responses with parameters adjusted for more natural endings
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Allowing for longer responses
            temperature=0.9,     # Higher temperature for more variability
            do_sample=True,
            top_p=0.92,          # Adjust nucleus sampling for broader output
            top_k=0,             # Not using top-k to allow broader sampling
            repetition_penalty=1.2,  # Applying a repetition penalty
            pad_token_id=tokenizer.pad_token_id,  # Using PAD token for padding if needed
            eos_token_id=None,   # Let the model decide when to end
        )

    # Decode the result
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the model's response
    response = result.split("[/INST]")[-1].strip()

    # Add logic to handle abrupt endings
    if response.endswith(tuple(['.', '!', '?'])) == False:
        # Assuming a sentence usually ends with . ! or ?
        response += '...'  # Adds an ellipsis to imply continuation

    return clean_response(response)

# Test the function
print(LLM_response("Hello! how are you doing?"))