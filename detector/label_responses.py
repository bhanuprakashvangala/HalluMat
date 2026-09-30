import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Load a BERT-based model
bert_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
model = AutoModel.from_pretrained(bert_model_name)

def get_embedding(text):
    """Generate BERT embeddings for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def classify_hallucination(response, factual_answer):
    """
    Compare responses at the token level and return similarity scores & hallucination levels.
    """
    if not response.strip() or not factual_answer.strip():
        return 0.0, "High"  # If empty, it's a hallucination

    response_embedding = get_embedding(response)
    factual_embedding = get_embedding(factual_answer)

    similarity_score = cosine_similarity([response_embedding], [factual_embedding])[0][0]

    if similarity_score > 0.8:
        level = "Low"
    elif 0.5 <= similarity_score <= 0.8:
        level = "Medium"
    else:
        level = "High"

    return similarity_score, level

# Load the dataset
file_path = DATA_DIR / "HalluMatData.csv"
df = pd.read_csv(file_path)

# Fill missing values
df["Generated_Response"] = df["Generated_Response"].fillna("")
df["Original_Answer"] = df["Original_Answer"].fillna("")

# Compute similarity scores and hallucination levels
df[["Hallucination_Score", "Hallucination_Level"]] = df.apply(
    lambda row: classify_hallucination(row["Generated_Response"], row["Original_Answer"]), axis=1, result_type="expand"
)

# Save or display results
df.to_csv("Hallucination_Analysis.csv", index=False)
print("Analysis complete. Results saved to 'Hallucination_Analysis.csv'.")
