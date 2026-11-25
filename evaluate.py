
import os
import csv
import torch
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from data_loader import test_loader, device
from model import create_model

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Store results
results = []


# Calculate evaluation metrics
def calculate_metrics(y_true, y_pred, model_name):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    print("\nModel:", model_name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })


# Evaluate PyTorch models
def evaluate_pytorch_model(model, model_name):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())

    calculate_metrics(y_true, y_pred, model_name)


# 1. Full Fine-Tuning
model = create_model()

model.load_state_dict(
    torch.load("outputs/full_best.pth", map_location=device)
)

model = model.to(device)

evaluate_pytorch_model(model, "Full Fine-Tuning")


# 2. Classifier Head
model = create_model()

model.load_state_dict(
    torch.load("outputs/head_best.pth", map_location=device)
)

model = model.to(device)

evaluate_pytorch_model(model, "Classifier Head")


# 3. Feature Extraction + SVM

# Load pretrained MobileNetV3
model = create_model()
model = model.to(device)
model.eval()

# Extract test features
features = []
labels = []

with torch.no_grad():

    for images, targets in test_loader:

        images = images.to(device)

        x = model.features(images)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)

        features.extend(x.cpu().numpy())
        labels.extend(targets.numpy())

# Load trained SVM
svm_model = joblib.load("outputs/svm_best.joblib")

# Predict test labels
svm_predictions = svm_model.predict(features)

calculate_metrics(
    labels,
    svm_predictions,
    "Feature Extraction + SVM"
)


# Save all results to CSV
csv_file = "outputs/evaluation_results.csv"

with open(csv_file, "w", newline="") as file:

    fieldnames = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

print("\nAll models evaluated successfully!")
print("CSV file saved at:", csv_file)
