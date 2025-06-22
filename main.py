import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for label in ["yes", "no"]:
            class_dir = os.path.join(root_dir, label)
            if os.path.exists(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append(
                            (os.path.join(class_dir, fname), 1 if label == "yes" else 0)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    device = get_device()

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def eval_model(model, loader, criterion):
    all_labels = []
    all_preds = []

    device = get_device()
    model.eval()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(imgs)
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds)

    predicted = [1 if p > 0.5 else 0 for p in all_preds]
    print(confusion_matrix(all_labels, predicted))
    print(classification_report(all_labels, predicted))
    roc_auc = roc_auc_score(all_labels, all_preds)
    print("ROC AUC:", roc_auc)
    return roc_auc


def main():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    device = get_device()
    print(f"Using device: {device}")

    full_dataset = BrainTumorDataset(
        "brain_tumor_dataset", transform=transform
    )

    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    from collections import Counter

    labels = [sample[1] for sample in full_dataset.samples]
    labels_counter = Counter(labels)
    print(labels_counter)  # Should be roughly balanced

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Binary classification
    model.to(device)


    yes = labels_counter[1]
    no = labels_counter[0]
    pos_weight = torch.tensor([no / yes],  dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    best_auc = 0
    epochs_no_improve = 0
    patience = 3  # Stop if no improvement for 3 epochs
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer)
        val_auc = eval_model(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{num_epochs}")

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_brain_tumor_resnet18.pth")
            print(f"Model improved. Saved model at epoch {epoch + 1}.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    torch.save(model.state_dict(), "brain_tumor_resnet18.pth")


if __name__ == "__main__":
    main()
