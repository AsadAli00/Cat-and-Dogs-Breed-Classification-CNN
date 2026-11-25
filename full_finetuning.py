
import time

from data_loader import train_loader, val_loader
from model import create_model
from train_model import train_model


# Load model
model = create_model()

# Enable training for all parameters
for param in model.parameters():
    param.requires_grad = True


# Count trainable parameters
trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", trainable_params)


# Start training
start_time = time.time()

train_model(
    model,
    train_loader,
    val_loader,
    "full"
)

end_time = time.time()

print(
    "Full fine-tuning time:",
    round(end_time - start_time, 2),
    "seconds"
)
