import cv2
import argparse
import numpy as np
from collections import deque
from fairface_model import FairFaceModel

class FaceTracker:
    def __init__(self, max_buffer_size=10, distance_threshold=100):
        """
        Initialize the FaceTracker.
        
        Args:
            max_buffer_size (int): How many past age predictions to keep for smoothing.
                                Larger = smoother but slower to react to changes.
            distance_threshold (int): Max pixels a face can move between frames to still be 
                                    considered the "same" face.
        """
        self.faces = []  # List to store currently tracked faces
        self.max_buffer_size = max_buffer_size
        self.distance_threshold = distance_threshold
        self.next_id = 0  # Unique ID counter for each new face

    def update(self, detected_faces):
        """
        Update tracked faces with new detections from the current frame.
        
        This function matches newly detected faces to existing tracked faces
        based on distance. It ensures we don't lose track of who is who
        even if the face moves slightly.
        """
        # 1. Format the new detections into a clean list of centers and data
        current_faces = []
        for face_data in detected_faces:
            region = face_data['region']
            # Calculate the center point (x, y) of the face
            center_x = region['x'] + region['w'] // 2
            center_y = region['y'] + region['h'] // 2
            current_faces.append({'center': (center_x, center_y), 'data': face_data})

        # 2. Match existing tracked faces to new detections
        matched_indices = set()
        
        for i, tracked in enumerate(self.faces):
            best_match_idx = -1
            min_dist = float('inf')
            
            # Find the closest face in the current frame to this tracked face
            for j, current in enumerate(current_faces):
                if j in matched_indices:
                    continue  # This face is already matched
                
                # Calculate Euclidean distance between old position and new position
                dist = np.sqrt((tracked['center'][0] - current['center'][0])**2 + 
                            (tracked['center'][1] - current['center'][1])**2)
                
                # If it's close enough, it might be a match
                if dist < min_dist and dist < self.distance_threshold:
                    min_dist = dist
                    best_match_idx = j
            
            # 3. Update the matched face with new data
            if best_match_idx != -1:
                matched_indices.add(best_match_idx)
                current = current_faces[best_match_idx]
                tracked['center'] = current['center']
                
                # Add new age prediction to the buffer
                tracked['age_buffer'].append(current['data']['age'])
                if len(tracked['age_buffer']) > self.max_buffer_size:
                    tracked['age_buffer'].popleft()
                
                # SMOOTHING LOGIC:
                # Instead of showing the instant prediction, show the most common
                # age from the last few frames. This prevents "flickering".
                from collections import Counter
                age_counts = Counter(tracked['age_buffer'])
                smoothed_age = age_counts.most_common(1)[0][0]
                
                current['data']['smoothed_age'] = smoothed_age
                tracked['last_seen_frame'] = 0  # Reset counter since we saw it
            else:
                # Face not found in this frame (maybe blocked or turned away)
                tracked['last_seen_frame'] += 1

        # 4. Remove faces we haven't seen for 5 frames
        self.faces = [f for f in self.faces if f.get('last_seen_frame', 0) < 5]

        # 5. Add any NEW faces that weren't matched to existing ones
        for j, current in enumerate(current_faces):
            if j not in matched_indices:
                new_face = {
                    'center': current['center'],
                    'age_buffer': deque([current['data']['age']], maxlen=self.max_buffer_size),
                    'id': self.next_id,
                    'last_seen_frame': 0
                }
                self.next_id += 1
                self.faces.append(new_face)
                # For a brand new face, smoothed age is just the current age
                current['data']['smoothed_age'] = current['data']['age']

        return [c['data'] for c in current_faces]

def main():
    # 1. Parse command line arguments
    # You can run with --image "photo.jpg" or just run without args for webcam
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to image file. Leave blank for webcam.')
    args = parser.parse_args()

    print("Loading FairFace model...")
    print("Using PyTorch for age, gender, and race detection...")
    
    try:
        model = FairFaceModel()
        print("FairFace model loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the FairFace model from:")
        print("https://drive.google.com/file/d/11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB/view")
        return

    # 2. Open Video Source
    # If args.image is present, load that file. Otherwise use webcam (0).
    video = cv2.VideoCapture(args.image if args.image else 0)
    tracker = FaceTracker()

    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break

        try:
            # 3. Analyze Frame
            # This single call does detection + alignment + prediction + emotion
            results = model.analyze(frame)
            
            if results:
                # 4. Update Tracker
                # Smoothes the age prediction so it doesn't jump around
                results = tracker.update(results)

                # 5. Draw Results
                for result in results:
                    region = result['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    
                    # Get the smoothed age if available, otherwise raw age
                    age = result.get('smoothed_age', result['age'])
                    gender = result['gender']
                    race = result['race']
                    emotion = result.get('emotion', 'neutral')
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Prepare labels
                    text1 = f"{gender}, {age}, {emotion}"
                    text2 = f"{race}"
                    
                    # Display labels above the face
                    cv2.putText(frame, text1, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, text2, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2, cv2.LINE_AA)

        except Exception as e:
            # Uncomment for debugging
            print(f"Error analyzing frame: {e}")
            pass

        # 6. Show the frame
        cv2.imshow("FairFace Analysis", frame)

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()