import cv2
import time
import numpy as np
import os

# Try to import YOLO
YOLO_AVAILABLE = False
yolo_model = None
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLO library imported successfully")
except ImportError as e:
    print(f"Warning: YOLO not available. Install ultralytics for advanced detection. Error: {e}")
except Exception as e:
    print(f"Warning: YOLO failed to initialize. Using basic detection only. Error: {e}")

# Simple FPS tracking
last_time = time.time()
frame_count = 0
fps = 0

# Initialize YOLO model (only if YOLO is available)
if YOLO_AVAILABLE:
    try:
        # Load YOLOv8 model (will download automatically on first run)
        print("Loading YOLO model...")
        yolo_model = YOLO('yolov8n.pt')  # nano model for speed, use 'yolov8s.pt' or 'yolov8m.pt' for better accuracy
        print("YOLO model loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Falling back to basic detection (faces and humans only)")
        YOLO_AVAILABLE = False
        yolo_model = None

# Initialize HOG descriptor for human detection (fallback)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize face cascade classifier
face_cascade = None
possible_paths = [
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
    'haarcascade_frontalface_default.xml'
]

for path in possible_paths:
    if os.path.exists(path):
        face_cascade = cv2.CascadeClassifier(path)
        if not face_cascade.empty():
            print(f"Face cascade loaded from: {path}")
            break

if face_cascade is None or face_cascade.empty():
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except:
        pass

# Object class mapping for YOLO COCO dataset
# YOLO can detect: vehicles, animals, and many other objects
CLASS_NAMES = {
    # Vehicles
    2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
    # Animals
    15: 'dog', 16: 'cat', 17: 'horse', 18: 'sheep', 19: 'cow',
    # Traffic related (we'll detect these with custom logic)
    # Traffic light is class 9 in COCO, but we'll enhance it
    9: 'traffic_light',
    # Person
    0: 'person',
    # Buffalo and Bullcart placeholders (mapping to similar objects or custom detection)
    19: 'cow', # Many models treat buffalo as cow or sheep
    17: 'horse', # Bullcarts often have cows/horses/donkeys
}

# Color mapping for different object types
COLOR_MAP = {
    'car': (255, 100, 0),           # Orange
    'motorcycle': (255, 150, 0),     # Light orange
    'bus': (255, 200, 0),            # Yellow-orange
    'truck': (255, 50, 0),           # Dark orange
    'traffic_light': (0, 255, 255),  # Cyan
    'person': (0, 255, 0),           # Green
    'dog': (255, 0, 255),            # Magenta
    'cat': (255, 100, 255),          # Light magenta
    'cow': (128, 0, 128),             # Purple
    'horse': (200, 0, 200),          # Light purple
    'sheep': (150, 0, 150),          # Medium purple
    'face': (255, 0, 0),              # Blue
    'zebra_crossing': (0, 255, 255),  # Cyan
    'footpath': (128, 128, 128),      # Gray
    'buffalo': (100, 50, 0),          # Brownish
    'bullock_cart': (200, 100, 0),    # Orange-brown
}

def detect_zebra_crossing(frame):
    """Detect zebra crossing using line detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    # Detect horizontal lines (zebra crossing stripes)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        horizontal_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is roughly horizontal
            if abs(y2 - y1) < 20 and abs(x2 - x1) > 50:
                horizontal_lines += 1
                cv2.line(frame, (x1, y1), (x2, y2), COLOR_MAP['zebra_crossing'], 2)
        
        return horizontal_lines >= 5  # If we find 5+ horizontal lines, likely a zebra crossing
    return False

def detect_footpath(frame):
    """Detect footpath/sidewalk using color and texture analysis"""
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Footpaths are typically gray/concrete colored
    # Define range for gray colors
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])
    
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    footpath_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Large enough to be a footpath
            x, y, w, h = cv2.boundingRect(contour)
            # Check if it's at the bottom of frame (typical footpath location)
            if y + h > frame.shape[0] * 0.6:
                cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_MAP['footpath'], 2)
                cv2.putText(frame, "Footpath", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MAP['footpath'], 2)
                footpath_detected = True
    
    return footpath_detected

def detect_objects(frame):
    """
    Detect multiple object types: vehicles, animals, traffic lights, zebra crossings, footpaths, faces, and humans.
    Returns: (frame, fps, counts_dict)
    """
    global last_time, frame_count, fps
    
    # Simple FPS calculation
    frame_count += 1
    current_time = time.time()
    if current_time - last_time >= 1.0:
        fps = frame_count
        frame_count = 0
        last_time = current_time
    
    # Initialize counts
    counts = {
        "Faces": 0,
        "Humans": 0,
        "Vehicles": 0,
        "Cars": 0,
        "Motorcycles": 0,
        "Buses": 0,
        "Trucks": 0,
        "Traffic_Lights": 0,
        "Dogs": 0,
        "Cats": 0,
        "Cows": 0,
        "Horses": 0,
        "Zebra_Crossings": 0,
        "Footpaths": 0,
        "Buffaloes": 0,
        "Bullock_Carts": 0
    }
    
    # Face detection (using Haar Cascade)
    face_count = 0
    if face_cascade is not None and not face_cascade.empty():
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            face_count = len(faces)
            counts["Faces"] = face_count
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_MAP['face'], 2)
                cv2.putText(frame, "Face", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MAP['face'], 2)
        except Exception as e:
            print(f"Face detection error: {e}")
    
    # YOLO detection for vehicles, animals, traffic lights, etc.
    if YOLO_AVAILABLE and yolo_model is not None:
        try:
            # Run YOLO inference
            results = yolo_model(frame, verbose=False, conf=0.25)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class and confidence
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Get class name
                    class_name = result.names[cls]
                    
                    # Map to our categories
                    if class_name in ['car']:
                        counts["Cars"] += 1
                        counts["Vehicles"] += 1
                        color = COLOR_MAP.get('car', (255, 100, 0))
                        label = f"Car {conf:.2f}"
                    elif class_name in ['motorcycle', 'motorbike']:
                        counts["Motorcycles"] += 1
                        counts["Vehicles"] += 1
                        color = COLOR_MAP.get('motorcycle', (255, 150, 0))
                        label = f"Motorcycle {conf:.2f}"
                    elif class_name in ['bus']:
                        counts["Buses"] += 1
                        counts["Vehicles"] += 1
                        color = COLOR_MAP.get('bus', (255, 200, 0))
                        label = f"Bus {conf:.2f}"
                    elif class_name in ['truck']:
                        counts["Trucks"] += 1
                        counts["Vehicles"] += 1
                        color = COLOR_MAP.get('truck', (255, 50, 0))
                        label = f"Truck {conf:.2f}"
                    elif class_name in ['traffic light', 'traffic_light']:
                        counts["Traffic_Lights"] += 1
                        color = COLOR_MAP.get('traffic_light', (0, 255, 255))
                        label = f"Traffic Light {conf:.2f}"
                    elif class_name in ['dog']:
                        counts["Dogs"] += 1
                        color = COLOR_MAP.get('dog', (255, 0, 255))
                        label = f"Dog {conf:.2f}"
                    elif class_name in ['cat']:
                        counts["Cats"] += 1
                        color = COLOR_MAP.get('cat', (255, 100, 255))
                        label = f"Cat {conf:.2f}"
                    elif class_name in ['cow']:
                        counts["Cows"] += 1
                        color = COLOR_MAP.get('cow', (128, 0, 128))
                        label = f"Cow {conf:.2f}"
                    elif class_name in ['horse']:
                        counts["Horses"] += 1
                        color = COLOR_MAP.get('horse', (200, 0, 200))
                        label = f"Horse {conf:.2f}"
                    elif class_name in ['person']:
                        counts["Humans"] += 1
                        color = COLOR_MAP.get('person', (0, 255, 0))
                        label = f"Person {conf:.2f}"
                    elif class_name in ['sheep']:
                        # Map sheep to buffalo for rural context if needed, or just track as buffalo
                        counts["Buffaloes"] += 1
                        color = COLOR_MAP.get('buffalo', (100, 50, 0))
                        label = f"Buffalo {conf:.2f}"
                    elif class_name in ['bicycle']:
                        # Bullock carts are slow moving, track them if detected as cart-like
                        # In standard COCO, bullock carts are rare, often detected as 'truck' or 'car'
                        # We'll add custom logic here or keep it simple
                        pass
                    else:
                        continue  # Skip other classes
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        except Exception as e:
            print(f"YOLO detection error: {e}")
    
    # Fallback to HOG for human detection if YOLO not available
    if not YOLO_AVAILABLE:
        try:
            result = hog.detectMultiScale(
                frame,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05,
                hitThreshold=0.0,
                finalThreshold=2.0
            )
            
            if isinstance(result, tuple):
                rects, weights = result
            else:
                rects = result
            
            if len(rects) > 0:
                rects = np.array(rects).reshape(-1, 4)
                counts["Humans"] = len(rects)
                
                for i, (x, y, w, h) in enumerate(rects):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_MAP['person'], 2)
                    cv2.putText(frame, "Human", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MAP['person'], 2)
        except Exception as e:
            print(f"Human detection error: {e}")
    
    # Detect zebra crossing
    if detect_zebra_crossing(frame):
        counts["Zebra_Crossings"] = 1
        cv2.putText(frame, "Zebra Crossing Detected", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_MAP['zebra_crossing'], 2)
    
    # Detect footpath
    if detect_footpath(frame):
        counts["Footpaths"] = 1
    
    return frame, fps, counts
