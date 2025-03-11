from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import os
import torch
from huggingface_hub import login
from peft import PeftModel, PeftConfig
from peft import AutoPeftModelForCausalLM



# Load environment variables from .env file
load_dotenv('.env')

# Get the token from environment variables
hf_token = os.getenv("HUGGING_FACE_TOKEN")

login(token=hf_token,add_to_git_credential=hf_token)

# Load the adapter config
peft_config = PeftConfig.from_pretrained("./fine_tuned_llama", token=hf_token)

# Load the base model specified in the adapter config

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_llama", token=hf_token)

# Load the adapter onto the base model
model = AutoModelForCausalLM.from_pretrained(
    "./fine_tuned_llama",
    token=hf_token
)
# model = PeftModel.from_pretrained(base_model, "./fine_tuned_llama", token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

test_input = tokenizer("Hello", return_tensors="pt")
with torch.no_grad():
    test_output = model(test_input.input_ids)
print("Model loaded successfully and can generate outputs")

# Function to generate responses
def generate_sql(instruction, context):
    prompt = f"### Instruction: {instruction}\n\n### Context: {context}"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Explicitly pass attention mask
            max_new_tokens=5000,
            num_return_sequences=1,
            temperature=0.2,
            max_time=120,
            pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id
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