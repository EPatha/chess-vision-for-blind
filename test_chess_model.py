#!/usr/bin/env python3
"""
Test trained chess detection model on test images or webcam

Usage:
    # Test on test dataset
    python3 test_chess_model.py --model runs/chess_detect/train/weights/best.pt --source test
    
    # Test on webcam
    python3 test_chess_model.py --model runs/chess_detect/train/weights/best.pt --source 0
    
    # Test on single image
    python3 test_chess_model.py --model runs/chess_detect/train/weights/best.pt --source image.jpg
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def test_model(
    model_path: str = "runs/chess_detect/train/weights/best.pt",
    source: str = "test",  # 'test', 0 (webcam), or image path
    conf: float = 0.25,
    iou: float = 0.45,
    save: bool = True,
    show: bool = False,
):
    """Test chess detection model."""
    
    print(f"🔍 Testing Chess Detection Model")
    print(f"   Model: {model_path}")
    print(f"   Source: {source}")
    print(f"   Confidence threshold: {conf}")
    print(f"   IoU threshold: {iou}")
    print("-" * 60)
    
    # Load trained model
    model = YOLO(model_path)
    
    # If source is 'test', use test dataset from data.yaml
    if source == "test":
        source_path = "Chess Pieces Detection Dataset/test/images"
    else:
        source_path = source
    
    # Run inference
    results = model.predict(
        source=source_path,
        conf=conf,
        iou=iou,
        save=save,
        show=show,
        stream=True if str(source_path) == "0" else False,
        verbose=True,
    )
    
    # Process results
    for i, result in enumerate(results):
        if hasattr(result, 'boxes'):
            boxes = result.boxes
            print(f"\nImage {i+1}: {len(boxes)} detections")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                cls_name = model.names[cls_id]
                print(f"  - {cls_name}: {conf_score:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Testing Complete!")
    if save:
        print(f"   Results saved to: runs/detect/predict/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 Chess Detection Model")
    parser.add_argument("--model", type=str, default="runs/chess_detect/train/weights/best.pt",
                        help="Path to trained model")
    parser.add_argument("--source", type=str, default="test",
                        help="Test source (test dataset, webcam=0, or image path)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save detection results")
    parser.add_argument("--show", action="store_true",
                        help="Show results in window")
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        show=args.show,
    )


if __name__ == "__main__":
    main()
