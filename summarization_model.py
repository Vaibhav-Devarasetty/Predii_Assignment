from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
# Disable high watermark in MPS for computational efficiency
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Using MPS Device as mac is Apple Silicon, you can change the device type to either "cuda" or "cpu"
device = torch.device("mps") 
torch.mps.empty_cache()

# Loading tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.float16).to(device)

# Setting pad_token_id to eos_token_id for handling padding
model.config.pad_token_id = model.config.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

# Input prompt
prompt = str(input())

# Tokenizing the input prompt
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
inputs['attention_mask'] = inputs['input_ids'].ne(tokenizer.pad_token_id).long()
# Moving the input tensors to the same device as the model
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generating response from the LLM Model
generate_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=30,
    pad_token_id=model.config.pad_token_id,
    eos_token_id=model.config.eos_token_id,
)

# Decoding and Printing the Generated Text
generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
print(generated_text)
