import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

hallumat_data = pd.read_csv(DATA_DIR / "HalluMatData.csv")

# Display column names to verify correct ones
print("Column Names:", hallumat_data.columns)

# Ensure correct column names are used (Update these if needed)
computed_column = "Hallucination_Level"
recomputed_column = "Computed_Hallucination_Level"

# Check if these columns exist in the dataset
if computed_column not in hallumat_data.columns or recomputed_column not in hallumat_data.columns:
    print(f"Error: One or both specified columns ({computed_column}, {recomputed_column}) are not in the dataset.")
else:
    # Map text labels to numerical values if necessary
    label_mapping = {"Low": 0, "Medium": 1, "High": 2}
    computed_labels = hallumat_data[computed_column].map(label_mapping)
    recomputed_labels = hallumat_data[recomputed_column].map(label_mapping)

    # Compute accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(computed_labels, recomputed_labels)
    precision, recall, f1_score, _ = precision_recall_fscore_support(computed_labels, recomputed_labels, average='macro')

    # Create a summary DataFrame
    evaluation_metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Score": [accuracy, precision, recall, f1_score]
    })

    # Display the results
    print(evaluation_metrics)

    # Save the table as a CSV file if needed
    evaluation_metrics.to_csv(Path(__file__).resolve().parents[1] / "results" / "agreement_metrics.csv", index=False)
