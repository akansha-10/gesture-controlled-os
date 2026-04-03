# Gesture Controlled OS

A real-time hand gesture recognition system that enables controlling operating system mouse actions using computer vision and machine learning.

The system uses a webcam feed to detect hand landmarks, classify hand gestures, and map them to OS-level mouse movements and click actions.

## Features
- Real-time hand landmark detection using MediaPipe
- Custom-trained machine learning model for hand pose classification
- Gesture-based mouse movement, clicks, scrolling, and dragging
- Smooth and stable cursor control using One Euro Filter
- Runs entirely locally with no cloud dependency

## Tech Stack
- Python
- OpenCV
- MediaPipe
- Keras / TensorFlow
- PyAutoGUI
- NumPy

## How It Works
1. Webcam captures live video frames
2. Hand landmarks are extracted using MediaPipe
3. Landmarks are normalized and passed to a trained ML model
4. Predicted hand gestures are mapped to mouse actions
5. Cursor movement is smoothed for stability and usability

## Project Structure
The main code lives inside the `hand_control/` folder.

## Running the Project

1. Install dependencies:

```bash
pip install -e .
```

2. Run the gesture controller:

```bash
python -m hand_control.cli.run --show
```

## Use Cases
- Touchless computer interaction
- Assistive technology
- Human-computer interaction research
- Computer vision experimentation

## Notes
- This project was built for learning and experimentation.
- Model weights and parameters can be retrained using the provided training scripts.
- Tested on Windows with a standard webcam.

## Author

Yash Nautiyal
