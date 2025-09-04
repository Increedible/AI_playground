# AI_playground

This repository is a collection of small AI and machine learning projects, mostly built from scratch. The focus is on experimenting with computer vision, neural networks, and training workflows, rather than providing polished end-user tools.

## Contents

### Core Neural Network Code
- **main.py**  
  A pure NumPy implementation of common neural network components (layers, activations, optimizers, losses, training loop). Used by several of the other projects.

- **pyTorchAI.py**  
  Similar concepts as `main.py`, but implemented with PyTorch for convenience and GPU support. Includes training code for classification tasks.

### Hand and Face Tracking
- **hand_tracking.py**  
  Uses MediaPipe Hands to either collect hand landmark data (for training) or run inference with a trained model to classify hand gestures (e.g. number of fingers raised).

- **hand_tracking_training.py**  
  Training script that takes landmark data from CSV files and trains a small neural network (using `main.py`) to classify gestures.

- **face_tracking.py**  
  Similar setup to hand tracking but applied to face landmarks. Can run in data collection mode (saving landmarks) or in analysis mode (predicting moods/labels with a trained model).

### Interactive Demo
- **ball_game.py**  
  A simple Pygame demo where you control a ball on the screen by moving your hand in front of the camera. Uses MediaPipe for hand tracking and applies physics to the ball.

### Text Model
- **bibleAI.py**  
  A character-level LSTM built with PyTorch to generate text. It reads plain text files (e.g. books in `Samples/Books/`), builds a vocabulary, trains a model, and can generate new sequences.

## How to Run
Most scripts can be run directly with Python:
- `python ball_game.py`
- `python hand_tracking.py`
- `python face_tracking.py`
- `python bibleAI.py`

Some scripts (like the tracking ones) will ask whether to run in **data collection** (`d`) or **analysis** (`a`) mode. Training scripts (`hand_tracking_training.py`, `pyTorchAI.py`, `bibleAI.py`) expect data files in the correct folders.

## Notes
- These are experimental projects, not production-ready code.  
- A webcam is required for the tracking demos.  
- Some scripts may save or load model files (`.model`, `.pth`, `.csv`) in local subfolders.  
