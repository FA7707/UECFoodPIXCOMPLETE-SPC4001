"""
Evaluate the food classifier on all 1000 test images.

Outputs:
  - Prints each prediction vs actual label
  - Separates results into correct and incorrect lists
  - Tallies per-class accuracy
  - Exports everything to evaluation_results.csv

Usage:
    python evaluate.py --data_dir ./UECFOODPIXCOMPLETE/data
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from food_classifier import (
    ONNX_PATH,
    MODEL_PATH,
    UECFoodDataset,
    create_npu_session,
    export_to_onnx,
    load_category_map,
    npu_predict,
)


def evaluate(data_dir: str):
    # Ensure ONNX model exists
    if not os.path.exists(ONNX_PATH):
        if os.path.exists(MODEL_PATH):
            print("ONNX model not found — exporting from PyTorch checkpoint...")
            export_to_onnx()
        else:
            print("Error: No model found. Train first with: python food_classifier.py --mode train")
            sys.exit(1)

    session = create_npu_session(ONNX_PATH)
    categories = load_category_map(str(Path(data_dir) / "category.txt"))
    test_ds = UECFoodDataset(data_dir, split="test", transform=None)

    correct_list = []   # (image_id, predicted_name, actual_name, confidence)
    incorrect_list = [] # (image_id, predicted_name, actual_name, confidence)
    all_results = []    # every result for CSV

    # Per-class tallies
    per_class_correct = {}
    per_class_total = {}

    print(f"\nEvaluating {len(test_ds)} test images...\n")

    for idx in tqdm(range(len(test_ds)), desc="Evaluating"):
        img_id = test_ds.image_ids[idx]
        true_label = test_ds.labels[idx]

        img_path = test_ds.img_dir / f"{img_id}.jpg"
        image = Image.open(img_path).convert("RGB")

        pred_label, probs = npu_predict(session, image)
        confidence = float(probs[pred_label]) * 100.0

        pred_name = categories.get(pred_label, f"class_{pred_label}")
        actual_name = categories.get(true_label, f"class_{true_label}")
        is_correct = pred_label == true_label

        # Per-class tally
        per_class_total[true_label] = per_class_total.get(true_label, 0) + 1
        per_class_correct[true_label] = per_class_correct.get(true_label, 0) + int(is_correct)

        row = {
            "image_id": img_id,
            "predicted_label": pred_label,
            "predicted_name": pred_name,
            "actual_label": true_label,
            "actual_name": actual_name,
            "correct": is_correct,
            "confidence": round(confidence, 2),
        }
        all_results.append(row)

        if is_correct:
            correct_list.append(row)
        else:
            incorrect_list.append(row)

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    total = len(all_results)
    num_correct = len(correct_list)
    num_incorrect = len(incorrect_list)

    print(f"\n{'='*70}")
    print(f"  CORRECT PREDICTIONS ({num_correct}/{total})")
    print(f"{'='*70}")
    for r in correct_list:
        print(f"  Image {r['image_id']:>6}  |  Predicted: {r['predicted_name']:<30}  |  Actual: {r['actual_name']:<30}  |  Conf: {r['confidence']:.1f}%")

    print(f"\n{'='*70}")
    print(f"  INCORRECT PREDICTIONS ({num_incorrect}/{total})")
    print(f"{'='*70}")
    for r in incorrect_list:
        print(f"  Image {r['image_id']:>6}  |  Predicted: {r['predicted_name']:<30}  |  Actual: {r['actual_name']:<30}  |  Conf: {r['confidence']:.1f}%")

    # -----------------------------------------------------------------------
    # Per-class tally
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  PER-CLASS ACCURACY")
    print(f"{'='*70}")
    print(f"  {'Class':<5} {'Name':<35} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*68}")
    for cls_id in sorted(per_class_total.keys()):
        name = categories.get(cls_id, f"class_{cls_id}")
        c = per_class_correct.get(cls_id, 0)
        t = per_class_total[cls_id]
        acc = 100.0 * c / t if t > 0 else 0.0
        print(f"  {cls_id:<5} {name:<35} {c:>8} {t:>8} {acc:>9.1f}%")

    # -----------------------------------------------------------------------
    # Overall summary
    # -----------------------------------------------------------------------
    overall_acc = 100.0 * num_correct / total if total > 0 else 0.0
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total images evaluated : {total}")
    print(f"  Correct                : {num_correct}")
    print(f"  Incorrect              : {num_incorrect}")
    print(f"  Overall accuracy       : {overall_acc:.2f}%")
    print(f"{'='*70}")

    # -----------------------------------------------------------------------
    # Export to CSV
    # -----------------------------------------------------------------------
    csv_path = "evaluation_results.csv"
    fieldnames = ["image_id", "predicted_label", "predicted_name",
                  "actual_label", "actual_name", "correct", "confidence"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Write correct predictions first
        for r in correct_list:
            writer.writerow(r)
        # Then incorrect predictions
        for r in incorrect_list:
            writer.writerow(r)

        # Blank separator row
        writer.writerow({k: "" for k in fieldnames})

        # Per-class summary rows
        summary_fields = ["image_id", "predicted_label", "predicted_name",
                          "actual_label", "actual_name", "correct", "confidence"]
        writer.writerow({
            "image_id": "CLASS_ID",
            "predicted_label": "CLASS_NAME",
            "predicted_name": "CORRECT",
            "actual_label": "TOTAL",
            "actual_name": "ACCURACY_%",
            "correct": "",
            "confidence": "",
        })
        for cls_id in sorted(per_class_total.keys()):
            name = categories.get(cls_id, f"class_{cls_id}")
            c = per_class_correct.get(cls_id, 0)
            t = per_class_total[cls_id]
            acc = 100.0 * c / t if t > 0 else 0.0
            writer.writerow({
                "image_id": cls_id,
                "predicted_label": name,
                "predicted_name": c,
                "actual_label": t,
                "actual_name": f"{acc:.2f}",
                "correct": "",
                "confidence": "",
            })

    print(f"\nResults exported to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate food classifier on all 1000 test images")
    parser.add_argument("--data_dir", type=str,
                        default="./UECFOODPIXCOMPLETE/data",
                        help="Path to UECFOODPIXCOMPLETE/data directory")
    args = parser.parse_args()
    evaluate(args.data_dir)


if __name__ == "__main__":
    main()
