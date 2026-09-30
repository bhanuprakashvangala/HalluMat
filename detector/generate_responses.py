from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os

# Read your Hugging Face token from the environment (export HF_TOKEN=...).

# Load the LLAMA model and tokenizer
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with the desired LLAMA model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=os.getenv("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=os.getenv("HF_TOKEN"))

# Set pad_token_id to eos_token_id to handle padding
tokenizer.pad_token_id = tokenizer.eos_token_id

# Function to generate responses
def generate_response(question, max_length=128, temperature=0.7, top_p=0.9):
    inputs = tokenizer(question, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id  # Avoid tokenization errors
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Load the paraphrased questions CSV
paraphrased_file = DATA_DIR / "Paraphrased_Questions.csv"
paraphrased_data = pd.read_csv(paraphrased_file)

# Ensure the dataset has the correct structure
paraphrased_data = paraphrased_data[["Question", "Answer", "Explanation"]]

# Generate responses for each paraphrased question
responses = []

for index, row in paraphrased_data.iterrows():
    question = row["Question"]
    try:
        response = generate_response(question)
        responses.append({
            "Question": question,
            "Generated_Response": response,
            "Original_Answer": row["Answer"],
            "Explanation": row["Explanation"]
        })
    except Exception as e:
        print(f"Error generating response for question at index {index}: {e}")
        continue

# Create a new DataFrame with generated responses
response_df = pd.DataFrame(responses)

# Save the results to a new CSV file
output_file = DATA_DIR / "Generated_Responses.csv"
response_df.to_csv(output_file, index=False)

# Display the first few rows of the generated responses
print(response_df.head())

print(f"Generated responses saved to {output_file}")
