import torch
import torch.nn.functional as F
from torch import nn


def accuracy(outputs, labels):
    """Top-1 accuracy"""
    _, preds = outputs.max(1)
    return (preds == labels).float().mean().item()


def validate(model, val_loader, device):
    """Run validation and return average accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total


def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """Knowledge Distillation Loss: α * soft loss + (1-α) * hard loss"""
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)

    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss


def save_checkpoint(model, path):
    """Save model checkpoint"""
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    """Load model weights from path"""
    model.load_state_dict(torch.load(path, map_location=device))
    return model