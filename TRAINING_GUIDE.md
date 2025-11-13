# Chess Vision Training & Testing Guide

## 📊 Current Status

### ✅ Training in Progress
- **Model**: YOLOv8n (3M parameters)
- **Dataset**: 606 train, 58 validation images
- **Classes**: 13 chess pieces (bishop, 6 black pieces, 6 white pieces)
- **Device**: Apple M1 (MPS GPU acceleration)
- **Epochs**: 50
- **Estimated Time**: 10-20 minutes

Training output location: `runs/chess_detect/train/`
Best model will be saved to: `runs/chess_detect/train/weights/best.pt`

---

## 🚀 After Training Completes

### 1. Evaluate Model Performance

Test on test dataset:
```bash
.venv/bin/python3 test_chess_model.py \
    --model runs/chess_detect/train/weights/best.pt \
    --source test \
    --conf 0.25 \
    --save
```

Results will show:
- Detection accuracy per class
- Confusion matrix
- Precision-Recall curves
- Sample predictions

### 2. Test Real-Time Detection with UI

Launch the UI (model path already configured):
```bash
.venv/bin/python3 yolov_ui.py
```

**UI Controls:**
- Source: `0` (built-in webcam) or `1` (external camera)
- Model: `runs/chess_detect/train/weights/best.pt` (pre-filled)
- Device: `cpu` or `mps` (M1 GPU)
- FPS: 10-30 (adjust based on performance)

Click **Start** and point camera at chess pieces!

### 3. Test with Single Image

```bash
.venv/bin/python3 test_chess_model.py \
    --model runs/chess_detect/train/weights/best.pt \
    --source path/to/chess_image.jpg \
    --conf 0.25 \
    --show
```

---

## 📈 Training Metrics to Monitor

Check training progress in terminal:
- **box_loss**: Bounding box localization loss (should decrease)
- **cls_loss**: Classification loss (should decrease)
- **mAP50**: Mean Average Precision @ IoU=0.5 (should increase, target >0.8)
- **mAP50-95**: mAP averaged over IoU 0.5-0.95 (should increase, target >0.5)

Training plots will be saved to:
- `runs/chess_detect/train/results.png` - Training curves
- `runs/chess_detect/train/confusion_matrix.png` - Class confusion matrix
- `runs/chess_detect/train/PR_curve.png` - Precision-Recall curve

---

## 🎯 Expected Results

### Good Model Performance:
- **mAP50** > 0.85 (85% accuracy at IoU threshold 0.5)
- **mAP50-95** > 0.65 (65% averaged across stricter IoU thresholds)
- Low confusion between similar pieces (e.g., bishop vs rook)

### If Performance is Low:
1. **Increase epochs**: Try 100-150 epochs
   ```bash
   .venv/bin/python3 train_chess_model.py --epochs 100 --batch 8 --device mps
   ```

2. **Use larger model**: Switch to yolov8s or yolov8m
   ```bash
   .venv/bin/python3 train_chess_model.py --model yolov8s.pt --epochs 100 --batch 8 --device mps
   ```

3. **Data augmentation**: Already configured with optimal settings (flip, scale, HSV)

---

## 🔧 Troubleshooting

### Training Crashes (Out of Memory)
- Reduce batch size: `--batch 4`
- Use smaller model: `--model yolov8n.pt`
- Switch to CPU: `--device cpu` (slower but more stable)

### Poor Detection on Webcam
- Adjust confidence threshold: Lower `conf` in UI (try 0.15-0.20)
- Ensure good lighting
- Camera should be ~30-50cm from chess pieces
- Avoid shadows and glare

### Model Not Found Error
- Wait for training to complete fully
- Check that `runs/chess_detect/train/weights/best.pt` exists
- Use absolute path if needed

---

## 📁 Project Structure

```
chess-vision-for-blind/
├── Chess Pieces Detection Dataset/
│   ├── train/images/          # 606 training images
│   ├── train/labels/          # YOLO format labels
│   ├── valid/images/          # 58 validation images
│   ├── valid/labels/
│   ├── test/images/           # Test set
│   ├── test/labels/
│   └── data.yaml              # Dataset configuration
├── runs/chess_detect/train/   # Training outputs
│   ├── weights/
│   │   ├── best.pt           # Best model (use this!)
│   │   └── last.pt           # Last epoch checkpoint
│   ├── results.png           # Training curves
│   ├── confusion_matrix.png
│   └── PR_curve.png
├── yolov_ui.py               # Real-time detection GUI
├── train_chess_model.py      # Training script
├── test_chess_model.py       # Testing/evaluation script
└── README.md                 # This file
```

---

## 🎓 Next Steps After Detection Works

1. **Chessboard Mapping**: Use `chess_board_utils.py` to map pieces to 8x8 grid
2. **Board State Recognition**: Track which piece is on which square
3. **Move Detection**: Detect when pieces move between squares
4. **Audio Feedback**: Add text-to-speech for blind users ("White pawn moves from E2 to E4")

---

## 💡 Tips for Best Results

1. **Lighting**: Consistent, diffuse lighting (avoid harsh shadows)
2. **Camera Angle**: Top-down or 45° angle works best
3. **Background**: Plain, contrasting background (not busy patterns)
4. **Piece Contrast**: Ensure black/white pieces are clearly visible
5. **Stable Mount**: Use tripod or stable surface for camera

---

## 📞 Common Commands Reference

```bash
# Monitor training progress
tail -f runs/chess_detect/train/results.txt

# Test model on test set
.venv/bin/python3 test_chess_model.py --model runs/chess_detect/train/weights/best.pt --source test

# Run real-time detection UI
.venv/bin/python3 yolov_ui.py

# Test on webcam without UI
.venv/bin/python3 test_chess_model.py --model runs/chess_detect/train/weights/best.pt --source 0 --show

# Re-train with more epochs
.venv/bin/python3 train_chess_model.py --epochs 100 --batch 8 --device mps
```

---

**Good luck with your training! Check back in 10-20 minutes.** 🚀♟️
