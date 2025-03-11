from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Load your model and tokenizer (do this once when the module is imported)
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fine_tuned_llama")

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("model and tokenizer loaded")
def generate_sql(instruction, context):
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
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

instruction = """Write a SQL query to find all endangered species with a population of less than 1000 individuals."""
context="""CREATE TABLE endangered_species (
    species_id INT PRIMARY KEY,
    scientific_name VARCHAR(100),
    common_name VARCHAR(100),
    conservation_status VARCHAR(50),
    habitat_type VARCHAR(100),
    population_count INT,
    year_assessed INT,
    primary_threats VARCHAR(200)"""

print(generate_sql(instruction,context))