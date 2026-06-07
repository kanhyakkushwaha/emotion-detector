# 🎭 Emotion Detector

A real-time AI-powered emotion detection system built with **Python**, **OpenCV**, **ONNX Runtime**, and **Transformer-based CLIP models**.

The application uses a webcam to detect faces and predict human emotions in real time. If the ONNX emotion model is unavailable, the system automatically falls back to a CLIP-based transformer model for emotion classification.

## Features

* Real-time webcam face detection
* Emotion recognition using deep learning models
* Automatic fallback from ONNX to CLIP
* OpenCV-based image processing
* Simple setup and execution
* Cross-platform Python implementation

## Tech Stack

* Python
* OpenCV
* NumPy
* ONNX Runtime
* PyTorch
* Hugging Face Transformers (CLIP)
* Pillow

## Project Structure

```text
emotion_detector.py                 # Main application
download_model.py                   # Model/resource downloader
emotion-ferplus-8.onnx              # Emotion recognition model
haarcascade_frontalface_default.xml # Face detection model
requirements.txt                    # Project dependencies
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kanhyakkushwaha/emotion-detector.git
cd emotion-detector
```

### 2. Create a Virtual Environment

```bash
python -m venv emotion_detect.venv
```

### 3. Activate the Environment

#### Windows PowerShell

```powershell
.\emotion_detect.venv\Scripts\Activate.ps1
```

#### Windows CMD

```cmd
emotion_detect.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python emotion_detector.py
```

## Controls

| Key | Action           |
| --- | ---------------- |
| Q   | Quit Application |
| ESC | Quit Application |

## How It Works

1. Captures video from the webcam.
2. Detects faces using Haar Cascade classifiers.
3. Attempts emotion prediction using the ONNX model.
4. Falls back to CLIP-based emotion classification if needed.
5. Displays predicted emotions directly on the video stream.

## Future Improvements

* Better emotion classification models
* GPU acceleration support
* Emotion analytics dashboard
* Model optimization for faster inference
* Multi-face emotion tracking

## Author

**Kanhya Kumar Kushwaha**

⭐ If you found this project useful, consider giving it a star.
