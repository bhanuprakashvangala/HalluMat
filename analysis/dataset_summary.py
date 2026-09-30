import pandas as pd

# Load your dataset
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

hallumat_data = pd.read_csv(DATA_DIR / "HalluMatData.csv")

# Each Group_ID is one original query; the rows in a group are its paraphrases
num_queries = hallumat_data["Group_ID"].nunique()

# Count the number of paraphrased versions (each query has paraphrased versions)
num_paraphrased_queries = hallumat_data["Question"].count()

# Count the number of generated responses
num_generated_responses = hallumat_data["Generated_Response"].count()

# Count the number of each hallucination category
hallucination_counts = hallumat_data["Hallucination_Level"].value_counts()

# Create a summary table
dataset_summary = pd.DataFrame({
    "Metric": ["Total Unique Queries", "Total Paraphrased Queries", "Total Generated Responses", 
               "Low Hallucination Responses", "Medium Hallucination Responses", "High Hallucination Responses"],
    "Count": [num_queries, num_paraphrased_queries, num_generated_responses, 
              hallucination_counts.get("Low", 0), hallucination_counts.get("Medium", 0), hallucination_counts.get("High", 0)]
})

# Display the table
print(dataset_summary)
