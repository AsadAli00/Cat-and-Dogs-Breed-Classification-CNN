
import time
import numpy as np
import torch
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

from data_loader import (
    train_loader,
    val_loader,
    test_loader
)

from model import create_model, device


# Load pretrained MobileNet
model = create_model()

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

model.eval()


# Function to extract features
def extract_features(loader):

    features_list = []
    labels_list = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            # Extract convolutional features
            features = model.features(images)

            # Average pooling
            features = model.avgpool(features)

            # Flatten features
            features = torch.flatten(features, 1)

            features_list.append(
                features.cpu().numpy()
            )

            labels_list.append(
                labels.numpy()
            )

    return (
        np.concatenate(features_list),
        np.concatenate(labels_list)
    )


# Extract training features
start_time = time.time()

X_train, y_train = extract_features(
    train_loader
)

X_val, y_val = extract_features(
    val_loader
)

X_test, y_test = extract_features(
    test_loader
)

feature_time = time.time() - start_time

print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Test features:", X_test.shape)


# Select SVM using validation macro-F1
best_model = None
best_f1 = -1
best_c = None

for c in [0.1, 1, 10]:

    svm_model = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=c)
    )

    svm_model.fit(X_train, y_train)

    val_predictions = svm_model.predict(X_val)

    val_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro"
    )

    print(
        f"C={c}, Validation Macro-F1={val_f1:.4f}"
    )

    if val_f1 > best_f1:

        best_f1 = val_f1
        best_model = svm_model
        best_c = c


# Save selected SVM model
joblib.dump(
    best_model,
    "outputs/svm_best.joblib"
)


# Evaluate on test set after model selection
start_time = time.time()

test_predictions = best_model.predict(X_test)

svm_training_time = time.time() - start_time

accuracy = accuracy_score(
    y_test,
    test_predictions
)

macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)

print("\nSVM Test Results")
print("----------------")
print("Selected C:", best_c)
print("Accuracy:", accuracy)
print("Macro-F1:", macro_f1)
print("Feature extraction time:", feature_time)
print("SVM test prediction time:", svm_training_time)
