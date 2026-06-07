import cv2
import numpy as np
import onnxruntime as ort
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def main():
    # Load the face cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Try to load the emotion recognition model
    session = None
    input_name = None
    emotions = []
    clip_model = None
    clip_processor = None
    try:
        session = ort.InferenceSession('emotion-ferplus-8.onnx')
        use_onnx = True
        input_name = session.get_inputs()[0].name
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        print("Emotion model loaded successfully.")
    except Exception as e:
        print("Using CLIP for real AI emotion detection.")
        use_onnx = False
        print(f"Warning: Could not load emotion model: {e}")
        print("Using CLIP for real AI emotion detection.")
        try:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("CLIP model loaded for emotion detection.")
        except Exception as e2:
            print(f"Warning: Could not load CLIP model: {e2}")
            print("Running with face detection and mock emotions.")
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Advanced Emotion Detector started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if session is not None:
                # Extract the face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                # Resize to 64x64
                face_resized = cv2.resize(face_roi, (64, 64))
                
                # Convert to RGB
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                
                # Normalize and prepare input
                input_data = face_rgb.astype(np.float32) / 255.0
                input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
                input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
                
                # Run inference
                outputs = session.run(None, {input_name: input_data})
                output = outputs[0]
                
                # Get the emotion with highest probability
                emotion_idx = np.argmax(output)
                emotion = emotions[emotion_idx]
                confidence = output[0][emotion_idx]
                
                # Put emotion text
                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            elif clip_model is not None:
                # Use CLIP for emotion detection
                face_roi = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (224, 224))  # CLIP expects 224x224
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(face_rgb)
                
                texts = ["a happy smiling face", "a sad face", "an angry face", "a surprised face", "a neutral face"]
                
                inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                emotion_idx = torch.argmax(probs).item()
                emotion = ["happy", "sad", "angry", "surprised", "neutral"][emotion_idx]
                confidence = probs[0][emotion_idx].item()
                
                # Put emotion text
                label = f"{emotion}: {confidence:.2f} (CLIP)"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                # No emotion model, use mock emotion for demonstration
                import random
                mock_emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral']
                emotion = random.choice(mock_emotions)
                confidence = random.uniform(0.5, 0.9)
                label = f"{emotion}: {confidence:.2f} (demo)"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Advanced Emotion Detector', frame)
        
        # Exit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is ESC key
            print("Stopping emotion detector...")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()