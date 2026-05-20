import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.config import (TRAIN_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)


# ─── Transforms 

def get_train_transforms() -> transforms.Compose:
   
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        # ImageNet mean/std — good default for transfer-style training
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─── Dataset Loaders 

def get_datasets() -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=get_train_transforms()
    )
    return train_dataset


def get_dataloaders(
    train_dataset: datasets.ImageFolder,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,               # Shuffle every epoch during training
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
   
    return train_loader


def verify_dataset(dataset: datasets.ImageFolder, name: str) -> None:
    print(f"\n{'─'*40}")
    print(f"  {name} Dataset")
    print(f"{'─'*40}")
    print(f"  Total samples : {len(dataset)}")
    print(f"  Classes found : {dataset.classes}")
    print(f"  Class → index : {dataset.class_to_idx}")
    print(f"{'─'*40}\n")