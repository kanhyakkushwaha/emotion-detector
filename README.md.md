# Emotion Detector - Setup Guide for Windows

This guide will help you set up and run the Emotion Detector project on Windows from scratch.

## Prerequisites

- Windows 10/11
- Python 3.9 or newer installed on your system
- PowerShell or CMD terminal
- A webcam (for running the detector)

## Project Structure

```
emotion_detector.py       - Main application (runs webcam emotion detection)
download_model.py         - Helper script to download model files
requirements.txt          - Python package dependencies
emotion.onnx             - Emotion recognition ONNX model (will be verified/downloaded)
haarcascade_frontalface_default.xml - Face detection cascade (will be verified/downloaded)
```

## Setup Steps

### Step 1: Create Virtual Environment

Open PowerShell in the project directory and run:

```powershell
python -m venv venv
```

### Step 2: Activate Virtual Environment

**PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Command Prompt (CMD):**
```cmd
venv\Scripts\activate.bat
```

> **Note:** If you get a permission error in PowerShell, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install:
- **opencv-python** - Computer vision and image processing
- **numpy** - Numerical computing
- **onnxruntime** - ONNX model inference engine
- **torch** - PyTorch for deep learning models
- **transformers** - Hugging Face transformers for CLIP model
- **pillow** - Image processing
- **requests** - HTTP library for downloading files

> **Installation Time:** This may take 5-15 minutes depending on your internet speed. PyTorch is large (~500-700 MB).

### Step 5: Verify/Download Model Files

**Option A: Run the download script (recommended for first-time setup)**

```powershell
python download_model.py
```

This will download:
- `haarcascade_frontalface_default.xml` - OpenCV face detection classifier
- `deploy.prototxt` and `deploy.caffemodel` - Backup emotion detection models

**Option B: Manual verification**

Check if `haarcascade_frontalface_default.xml` exist in the project folder. If missing:
- `haarcascade_frontalface_default.xml` will be downloaded by `download_model.py`
- `emotion.onnx` should already be in your project folder, or the app will fall back to CLIP model

### Step 6: Run the Application

```powershell
python emotion_detector.py
```

### Controls

- **Press `q`** - Quit the application
- **Press `ESC`** - Alternative quit command

## Deactivating Virtual Environment

When finished, deactivate the virtual environment:

```powershell
deactivate
```

## Troubleshooting

### Issue: "ModuleNotFoundError" when running the app

**Solution:** Make sure the virtual environment is activated. You should see `(venv)` at the start of your terminal prompt.

### Issue: Webcam not found

**Solution:** 
- Check if another application is using the webcam
- Check Device Manager to ensure camera is enabled
- Try restarting the application

### Issue: ONNX model not found

**Solution:** The app is designed to handle this gracefully. It will fall back to using the CLIP model for emotion detection, which will be downloaded automatically on first use.

### Issue: Slow performance or high CPU usage

**Solution:**
- This is normal as the models are large. CLIP model is particularly heavy.
- PyTorch/CUDA GPU acceleration may help if you have a compatible GPU
- Try closing other applications to free up memory

### Issue: Model downloads fail

**Solution:** 
- Check your internet connection
- Run `python download_model.py` again
- Manually download from:
  - Face cascade: https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml
  - And place it in the project folder

## Quick Start (Copy-Paste Commands)

For experienced users, here's the complete setup in one code block (PowerShell):

```powershell
# Create and activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies (takes 5-15 minutes)
pip install -r requirements.txt

# Download model files
python download_model.py

# Run the app
python emotion_detector.py
```

## Additional Notes

- **Model Files:** The app supports multiple fallback models:
  1. ONNX model (`emotion.onnx`) - Fastest
  2. CLIP model (auto-downloaded) - More accurate
  3. Mock emotions - For testing without models

- **GPU Acceleration:** If you have an NVIDIA GPU, PyTorch will automatically use CUDA for faster inference. No additional setup needed.

- **Virtual Environment Size:** The venv + dependencies will take ~2-3 GB of disk space.

## Getting Help

If you encounter issues:
1. Check the error message carefully - most are self-explanatory
2. Ensure all prerequisites are met (Python 3.9+, pip updated)
3. Try reinstalling dependencies: `pip install -r requirements.txt --force-reinstall`
4. Check that all resource files exist in the project directory

---

**Ready?** Start with Step 1 above!
