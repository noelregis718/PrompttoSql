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

login(token=hf_token)

# Load the adapter config
peft_config = PeftConfig.from_pretrained("./fine_tuned_llama", token=hf_token)

# Load the base model specified in the adapter config
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    device_map="auto",
    trust_remote_code=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_llama", token=hf_token)

# Load the adapter onto the base model
model = AutoPeftModelForCausalLM.from_pretrained(
    "./fine_tuned_llama",
    token=hf_token
)
# model = PeftModel.from_pretrained(base_model, "./fine_tuned_llama", token=hf_token)

# Create the pipeline
generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True 
)

# Function to generate responses
def generate_sql(instruction, context):
    prompt = f"### Instruction: {instruction}\n\n### Context: {context}"
    response = generator(prompt, max_length=500, num_return_sequences=1)
    return response[0]['generated_text']

instruction = 'Write a SQL query to find all endangered species with a population of less than 1000 individuals.'
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