from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import gc

# Global variables to store model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fine_tuned_deepseek")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Model and tokenizer loaded")

def unload_model():
    global model, tokenizer
    del model
    del tokenizer
    model = None
    tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model and tokenizer unloaded")

def generate_sql(instruction, context):
    global model, tokenizer
    
    # Load model if not already loaded
    load_model()
    
    prompt = f"### Instruction: {instruction}\n\n### Context: {context}"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1000,
            temperature=0.2,
            max_time=120
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Optionally unload model after generation
    # Uncomment if you want to free memory after each generation
    # unload_model()
    
    return result
