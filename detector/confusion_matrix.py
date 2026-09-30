import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset 
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

hallumat_data = pd.read_csv(DATA_DIR / "HalluMatData.csv")

# Ensure column names match dataset structure
# Assuming `Computed_Hallucination_Level` and `Recomputed_Hallucination_Level` exist in the dataset
computed_labels = hallumat_data["Hallucination_Level"]
recomputed_labels = hallumat_data["Computed_Hallucination_Level"]

# Mapping text labels to numerical values if needed (modify if labels are different)
label_mapping = {"Low": 0, "Medium": 1, "High": 2}
computed_labels = computed_labels.map(label_mapping)
recomputed_labels = recomputed_labels.map(label_mapping)

# Compute Confusion Matrix
cm = confusion_matrix(computed_labels, recomputed_labels)
labels = ["Low", "Medium", "High"]

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Recomputed Labels")
plt.ylabel("Computed Labels")
plt.title("Confusion Matrix: Computed vs Recomputed Hallucination Levels")
plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
