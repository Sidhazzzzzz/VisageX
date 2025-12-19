import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
import os
from fer.fer import FER

class FairFaceModel:
    def __init__(self, model_path='res34_fair_align_multi_7_20190809.pt'):
        """
        Initialize the FairFace Model.
        
        Args:
            model_path (str): Path to the pre-trained weights file.
        """
        # Select the hardware to run on (Graphics Card if available, else CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 1. Face Detector: Uses Haar Cascades (fast, lightweight) to find faces
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 2. Emotion Detector: Uses FER library
        # mtcnn=False means we use our own lighter detection to keep it fast
        self.emotion_detector = FER(mtcnn=False)
        
        # 3. Main AI Model: ResNet34
        # We replace the final layer (fc) to output exactly 18 numbers:
        # 7 for Race + 2 for Gender + 9 for Age
        self.model = models.resnet34(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 18)
        
        # Load weights
        model_full_path = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(model_full_path):
            self.model.load_state_dict(torch.load(model_full_path, map_location=self.device, weights_only=True))
        else:
            raise FileNotFoundError(f"Model file not found: {model_full_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        
        # 4. Image Preprocessing
        # Neural networks expect images in a very specific format.
        # This pipeline converts our raw camera image into that format.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),        # Convert numpy array to PIL Image
            transforms.Resize((224, 224)),  # Resize to 224x224 (required by ResNet)
            transforms.ToTensor(),          # Convert to Tensor (math matrix)
            # Normalize color values using standard ImageNet stats
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Labels
        self.race_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
        self.gender_labels = ['Male', 'Female']
        self.age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    
    def detect_faces(self, frame):
        """
        Scan the entire frame to find faces.
        Returns a list of rectangles: (x, y, width, height)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # scaleFactor=1.1: Image is reduced by 10% at each scale
        # minNeighbors=5: Higher quality threshold (reduces false positives)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    
    def align_face(self, frame, face_rect):
        """
        Cut out the face from the main picture with a little extra margin.
        The AI works better if it sees the whole head, not just the features.
        """
        x, y, w, h = face_rect
        
        # Calculate padding (25% of the size)
        pad = int(0.25 * max(h, w))
        
        # Calculate coordinates making sure we don't go off the edge of the image
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        
        # Crop the image
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None
            
        # Convert BGR to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        return face_img
    
    def predict_emotion(self, frame, face_rect):
        """
        Assess the emotion of the face using the FER library.
        Returns: emotion_name (str), confidence_score (float)
        """
        x, y, w, h = face_rect
        
        # Add some padding for better emotion detection (needs context)
        pad = int(0.1 * max(h, w))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        
        face_region = frame[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return 'neutral', 0.0
        
        try:
            # Detect emotion in the face region
            emotions = self.emotion_detector.detect_emotions(face_region)
            if emotions and len(emotions) > 0:
                # FER returns a list of emotions with scores, pick the highest
                emotion_scores = emotions[0]['emotions']
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[dominant_emotion]
                return dominant_emotion, confidence
        except Exception:
            pass
        
        return 'neutral', 0.0
    
    def predict(self, face_img):
        """
        The brain of the operation. Takes a face image and outputs attributes.
        """
        if face_img is None or face_img.size == 0:
            return None
        
        # 1. Prepare image (Resize -> CSS Normalization -> Tensor)
        img_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
        
        # 2. Run the AI Model
        with torch.no_grad():
            outputs = self.model(img_tensor)
            outputs = outputs.cpu().numpy().squeeze()
        
        # 3. Split the 18 outputs into their categories
        # Indices 0-6: Race
        # Indices 7-8: Gender
        # Indices 9-17: Age
        race_scores = outputs[:7]
        gender_scores = outputs[7:9]
        age_scores = outputs[9:18]
        
        # 4. Convert raw scores to concrete predictions
        # We use softmax/exp to convert scores to "probabilities"
        # Then we pick the one with the highest probability (argmax)
        
        race_probs = np.exp(race_scores) / np.sum(np.exp(race_scores))
        gender_probs = np.exp(gender_scores) / np.sum(np.exp(gender_scores))
        age_probs = np.exp(age_scores) / np.sum(np.exp(age_scores))
        
        race_idx = np.argmax(race_probs)
        gender_idx = np.argmax(gender_probs)
        age_idx = np.argmax(age_probs)
        
        return {
            'race': self.race_labels[race_idx],
            'race_confidence': float(race_probs[race_idx]),
            'gender': self.gender_labels[gender_idx],
            'gender_confidence': float(gender_probs[gender_idx]),
            'age': self.age_labels[age_idx],
            'age_confidence': float(age_probs[age_idx])
        }
    
    def analyze(self, frame):
        """
        Orchestrator function:
        1. Finds all faces
        2. Aligns/Crops them
        3. Predicts Attributes (Age, Gender, Race)
        4. Predicts Emotion
        5. Returns everything in a nice list
        """
        faces = self.detect_faces(frame)
        results = []
        
        for face in faces:
            x, y, w, h = face
            face_img = self.align_face(frame, face)
            
            # Run the heavy AI model
            prediction = self.predict(face_img)
            
            if prediction:
                # Run the emotion detector
                emotion, emotion_conf = self.predict_emotion(frame, face)
                prediction['emotion'] = emotion
                prediction['emotion_confidence'] = emotion_conf
                
                # Store the location so we can draw the box later
                prediction['region'] = {
                    'x': int(x),
                    'y': int(y),
                    'w': int(w),
                    'h': int(h)
                }
                results.append(prediction)
        
        return results
