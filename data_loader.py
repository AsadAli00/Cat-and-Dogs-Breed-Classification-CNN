
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
weights = models.MobileNet_V3_Small_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = weights.transforms()

# Load Oxford-IIIT Pet dataset
train_dataset = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=True,
    transform=train_transform
)

test_dataset = datasets.OxfordIIITPet(
    root="./data",
    split="test",
    target_types="category",
    download=True,
    transform=test_transform
)

# Get labels for stratified splitting
labels = [label for _, label in train_dataset]

# Split training data into train and validation
train_indices, val_indices = train_test_split(
    range(len(labels)),
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Create separate datasets
train_subset = Subset(train_dataset, train_indices)

# Use non-augmented transformations for validation
val_dataset = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=False,
    transform=test_transform
)

val_subset = Subset(val_dataset, val_indices)

# DataLoaders
train_loader = DataLoader(
    train_subset, batch_size=32, shuffle=True
)

val_loader = DataLoader(
    val_subset, batch_size=32, shuffle=False
)

test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False
)


class_names = train_dataset.classes

print("Training images:", len(train_subset))
print("Validation images:", len(val_subset))
print("Testing images:", len(test_dataset))
print("Number of classes:", len(train_dataset.classes))
