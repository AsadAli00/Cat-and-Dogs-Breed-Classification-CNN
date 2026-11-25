
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score

from model import device

import os
os.makedirs("outputs", exist_ok=True)

EPOCHS = 10
LEARNING_RATE = 0.0001


def train_model(
    model,
    train_loader,
    val_loader,
    approach
):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE
    )

    best_f1 = 0.0

    for epoch in range(EPOCHS):

        # Training
        model.train()

        # Keep frozen feature extractor in eval mode
        if approach == "head":
            model.features.eval()

        running_loss = 0.0

        for step, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], "
            f"Training Loss: "
            f"{running_loss / len(train_loader):.4f}"
        )

        # Validation
        model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(
                    predicted.cpu().numpy()
                )

                all_labels.extend(
                    labels.numpy()
                )

        val_f1 = f1_score(
            all_labels,
            all_predictions,
            average="macro"
        )

        print(
            f"Validation Macro-F1: {val_f1:.4f}"
        )

        # Save best model
        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(
                model.state_dict(),
                f"outputs/{approach}_best.pth"
            )

            print("Best model saved.")

    print("Training completed.")
