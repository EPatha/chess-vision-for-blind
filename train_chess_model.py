#!/usr/bin/env python3
"""
Train YOLOv8 model on Chess Pieces Detection Dataset

Usage:
    python3 train_chess_model.py --epochs 100 --imgsz 640 --batch 16

Output:
    runs/detect/train/weights/best.pt - Best model weights
    runs/detect/train/weights/last.pt - Last epoch weights
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_chess_model(
    data_yaml: str = "Chess Pieces Detection Dataset/data.yaml",
    model: str = "yolov8n.pt",  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",  # cpu or cuda or mps (Apple Silicon)
    project: str = "runs/chess_detect",
    name: str = "train",
):
    """Train YOLOv8 on chess pieces dataset."""
    
    print(f"🚀 Starting YOLOv8 Chess Detection Training")
    print(f"   Model: {model}")
    print(f"   Data: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch}")
    print(f"   Device: {device}")
    print("-" * 60)
    
    # Fix data.yaml path issue (update to absolute path)
    data_path = Path(data_yaml).resolve()
    
    # Load pretrained model
    model = YOLO(model)
    
    # Train the model
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=20,  # early stopping patience
        save=True,
        save_period=10,  # save checkpoint every 10 epochs
        plots=True,  # save training plots
        val=True,  # validate during training
        # Augmentation settings
        hsv_h=0.015,  # image HSV-Hue augmentation
        hsv_s=0.7,    # image HSV-Saturation augmentation
        hsv_v=0.4,    # image HSV-Value augmentation
        degrees=0.0,  # image rotation (+/- deg)
        translate=0.1,  # image translation (+/- fraction)
        scale=0.5,    # image scale (+/- gain)
        flipud=0.0,   # image flip up-down (probability)
        fliplr=0.5,   # image flip left-right (probability)
        mosaic=1.0,   # image mosaic (probability)
    )
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"   Best model saved to: {project}/{name}/weights/best.pt")
    print(f"   Last model saved to: {project}/{name}/weights/last.pt")
    print(f"   Results saved to: {project}/{name}/")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Chess Detection Dataset")
    parser.add_argument("--data", type=str, default="Chess Pieces Detection Dataset/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model variant (yolov8n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for training")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (reduce if out of memory)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--project", type=str, default="runs/chess_detect",
                        help="Project directory")
    parser.add_argument("--name", type=str, default="train",
                        help="Experiment name")
    
    args = parser.parse_args()
    
    train_chess_model(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
