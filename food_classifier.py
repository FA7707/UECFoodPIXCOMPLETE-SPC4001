"""
Food Image Classifier using the UECFoodPixComplete dataset.

Trains a ResNet-50 model to classify food images into 103 categories (0=background + 102 food types).
Supports Windows 11 NPU acceleration via ONNX Runtime DirectML.

Usage:
    # Train the model (uses GPU/CPU via PyTorch)
    python food_classifier.py --mode train --data_dir ./UECFOODPIXCOMPLTE/data/UECFoodPIXCOMPLTE

    # Evaluate on test set using NPU (DirectML)
    python food_classifier.py --mode test --data_dir ./UECFOODPIXCOMPLTE/data/UECFoodPIXCOMPLTE

    # Classify a single image using NPU
    python food_classifier.py --mode predict --image path/to/image.jpg

    # Export trained model to ONNX (for NPU inference)
    python food_classifier.py --mode export
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_category_map(category_file: str) -> dict[int, str]:
    """Load category.txt → {id: name} mapping."""
    categories = {}
    with open(category_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("id"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                categories[int(parts[0])] = parts[1]
    return categories


def get_image_label_from_mask(mask_path: str) -> int:
    """Determine the dominant food label from a segmentation mask.

    Each pixel's R channel value equals the class id.  We find the most
    frequent non-zero (non-background) class.
    """
    mask = np.array(Image.open(mask_path).convert("RGB"))
    r_channel = mask[:, :, 0]
    non_bg = r_channel[r_channel > 0]
    if len(non_bg) == 0:
        return 0  # background only
    values, counts = np.unique(non_bg, return_counts=True)
    return int(values[counts.argmax()])


class UECFoodDataset(Dataset):
    """PyTorch dataset for UECFoodPixComplete."""

    def __init__(self, data_dir: str, split: str = "train", transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Read image ids from train.txt / test.txt
        split_file = self.data_dir / f"{split}.txt"
        with open(split_file, "r") as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        self.img_dir = self.data_dir / split / "img"
        self.mask_dir = self.data_dir / split / "mask"

        # Pre-compute labels from masks
        print(f"Loading {split} labels from masks...")
        self.labels = []
        valid_ids = []
        for img_id in tqdm(self.image_ids, desc=f"Scanning {split} masks"):
            mask_path = self.mask_dir / f"{img_id}.png"
            img_path = self.img_dir / f"{img_id}.jpg"
            if mask_path.exists() and img_path.exists():
                label = get_image_label_from_mask(str(mask_path))
                self.labels.append(label)
                valid_ids.append(img_id)
        self.image_ids = valid_ids
        print(f"  Found {len(self.image_ids)} valid {split} samples.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.img_dir / f"{img_id}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 103  # 0 (background) + 102 food categories

MODEL_PATH = "food_classifier.pth"
ONNX_PATH = "food_classifier.onnx"


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build a ResNet-50 model with a custom classification head."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    data_dir = args.data_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_ds = UECFoodDataset(data_dir, split="train", transform=TRAIN_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{100.0 * correct / total:.1f}%")

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch}: loss={epoch_loss:.4f}  acc={epoch_acc:.1f}%")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Saved best model to {MODEL_PATH}")

    print("Training complete.")
    # Auto-export to ONNX after training
    export_to_onnx(model)


def export_to_onnx(model: nn.Module = None):
    """Export the trained PyTorch model to ONNX format for NPU inference."""
    if model is None:
        model = build_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    model.eval().cpu()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {ONNX_PATH}")


# ---------------------------------------------------------------------------
# NPU Inference (ONNX Runtime + DirectML)
# ---------------------------------------------------------------------------

def create_npu_session(onnx_path: str):
    """Create an ONNX Runtime session with DirectML (NPU/GPU) provider."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    print(f"Available ONNX Runtime providers: {available}")

    # DirectML enables NPU/GPU acceleration on Windows
    if "DmlExecutionProvider" in available:
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        print("Using DirectML (NPU/GPU) for inference.")
    else:
        providers = ["CPUExecutionProvider"]
        print("DirectML not available — falling back to CPU inference.")
        print("Install onnxruntime-directml for NPU support.")

    return ort.InferenceSession(onnx_path, providers=providers)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Apply test-time preprocessing and return a numpy array."""
    tensor = TEST_TRANSFORM(image.convert("RGB"))
    return tensor.unsqueeze(0).numpy()


def npu_predict(session, image: Image.Image) -> tuple[int, np.ndarray]:
    """Run a single image through the ONNX model and return (class_id, probabilities)."""
    input_data = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    logits = outputs[0][0]
    # softmax
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return int(np.argmax(probs)), probs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test(args):
    """Evaluate the model on the test set using NPU via ONNX Runtime."""
    data_dir = args.data_dir

    if not os.path.exists(ONNX_PATH):
        if os.path.exists(MODEL_PATH):
            print("ONNX model not found — exporting from PyTorch checkpoint...")
            export_to_onnx()
        else:
            print(f"Error: No model found. Train first with --mode train")
            sys.exit(1)

    session = create_npu_session(ONNX_PATH)
    categories = load_category_map(str(Path(data_dir) / "category.txt"))

    test_ds = UECFoodDataset(data_dir, split="test", transform=None)

    correct = 0
    total = 0
    top5_correct = 0
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}

    print(f"\nEvaluating on {len(test_ds)} test images...\n")
    for idx in tqdm(range(len(test_ds)), desc="Testing"):
        img_id = test_ds.image_ids[idx]
        true_label = test_ds.labels[idx]

        img_path = test_ds.img_dir / f"{img_id}.jpg"
        image = Image.open(img_path).convert("RGB")

        pred_label, probs = npu_predict(session, image)

        # Top-1
        is_correct = pred_label == true_label
        correct += int(is_correct)
        total += 1

        # Top-5
        top5 = np.argsort(probs)[-5:][::-1]
        top5_correct += int(true_label in top5)

        # Per-class tracking
        per_class_total[true_label] = per_class_total.get(true_label, 0) + 1
        per_class_correct[true_label] = per_class_correct.get(true_label, 0) + int(is_correct)

    top1_acc = 100.0 * correct / total
    top5_acc = 100.0 * top5_correct / total
    print(f"\n{'='*60}")
    print(f"  Test Results")
    print(f"{'='*60}")
    print(f"  Total images : {total}")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"{'='*60}")

    # Per-class breakdown
    print(f"\n{'Class':<5} {'Name':<35} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("-" * 70)
    for cls_id in sorted(per_class_total.keys()):
        name = categories.get(cls_id, f"class_{cls_id}")
        c = per_class_correct.get(cls_id, 0)
        t = per_class_total[cls_id]
        acc = 100.0 * c / t if t > 0 else 0.0
        print(f"{cls_id:<5} {name:<35} {c:>8} {t:>8} {acc:>7.1f}%")


# ---------------------------------------------------------------------------
# Single-image prediction
# ---------------------------------------------------------------------------

def predict(args):
    """Classify a single image using NPU inference."""
    if not os.path.exists(ONNX_PATH):
        if os.path.exists(MODEL_PATH):
            export_to_onnx()
        else:
            print(f"Error: No model found. Train first with --mode train")
            sys.exit(1)

    session = create_npu_session(ONNX_PATH)

    # Try to load category names
    cat_file = Path(args.data_dir) / "category.txt" if args.data_dir else None
    categories = {}
    if cat_file and cat_file.exists():
        categories = load_category_map(str(cat_file))

    image = Image.open(args.image).convert("RGB")
    pred_label, probs = npu_predict(session, image)

    print(f"\nImage: {args.image}")
    print(f"Predicted class: {pred_label} — {categories.get(pred_label, 'unknown')}")
    print(f"Confidence: {probs[pred_label]*100:.1f}%")

    print("\nTop-5 predictions:")
    top5 = np.argsort(probs)[-5:][::-1]
    for rank, cls_id in enumerate(top5, 1):
        name = categories.get(cls_id, f"class_{cls_id}")
        print(f"  {rank}. {name} (class {cls_id}) — {probs[cls_id]*100:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Food Image Classifier (UECFoodPixComplete)")
    parser.add_argument("--mode", choices=["train", "test", "predict", "export"],
                        required=True, help="Operation mode")
    parser.add_argument("--data_dir", type=str,
                        default="./UECFOODPIXCOMPLTE/data/UECFoodPIXCOMPLTE",
                        help="Path to UECFoodPIXCOMPLTE data directory")
    parser.add_argument("--image", type=str, help="Path to image (predict mode)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "predict":
        if not args.image:
            parser.error("--image is required for predict mode")
        predict(args)
    elif args.mode == "export":
        export_to_onnx()


if __name__ == "__main__":
    main()
