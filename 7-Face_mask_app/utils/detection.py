import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Model path
model_path = r"E:/Omar/Courses/AMIT-main/Final Project of AI/PLAN D (Face Mask Detection)/MobileNetV2.h5"

# Load the model
def load_mask_model():
    
    model = load_model(model_path)
    return model

# Load the pre-trained Caffe model for face detection
# 1. prototxt: Defines the model architecture (layers)
# 2. caffemodel: Contains the trained weights
face_net = cv2.dnn.readNetFromCaffe(
    r"E:/Omar/Courses/AMIT-main/Final Project of AI/PLAN D (Face Mask Detection)/deploy.prototxt",
    r"E:/Omar/Courses/AMIT-main/Final Project of AI/PLAN D (Face Mask Detection)/res10_300x300_ssd_iter_140000.caffemodel"
)

# Definition of a function
"""
The function takes:
- model → Mask detection model (Keras)
- frame → Camera image
- threshold → Threshold separating Mask / No Mask
"""
def detect_mask_dnn(model, frame, threshold=0.5):
    
    # Extracting frame dimensions ((height, width, channels))
    # It is later used to convert DNN coordinates from normalized to actual pixels.
    h, w = frame.shape[:2]
    
    # It converts the image (frame) into a blob, which is the format needed by the DNN (Caffe Face Detection Network) model.
    # blob ==> It is an image that has been: Resized, Reordered, Meaned (subtracted) and Prepared to feed into the neural model, So that the input is ready for use within.
    """
    frame ==> Camera image
    1.0 ==> scalefactor
    (300, 300) ==> This is the required image size for the model.
    (104.0, 177.0, 123.0) ==> (BGR) These are the Mean subtraction values. The Caffe network was trained on data to which the same process was applied.
    """
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # Give the model the picture that he will analyze.
    face_net.setInput(blob)
    
    # This is the line that actually activates the Face Detector.
    # detections = [Batch Size, Class Label, Detections, Data Info]
    detections = face_net.forward()
    
    # Create a list to save results, All detected faces will be stored there along with:
    # - The box
    # - The label (With Mask / Without Mask)
    # - The confidence rating
    results = []
    """{ 
    'box': (x, y, width, height), 
    'label': "With Mask" OR "Without Mask", 
    'confidence': a number between 0 and 1
    }
    """
    
    # Passing over each detected face
    # detections.shape[2] = Number of faces detected.
    for i in range(detections.shape[2]):
        
        # Caffe models like res10_300x300_ssd when you forward, return the result in the form of a (4D Array), like this: [Batch Size, Class Label, Detections, Data Info]
        """ 
        Why [0, 0, i, 2]?
        
        - The first zero [0]: This is the batch number. Since we're working on one image (or frame) at a time, we always select image number 0 (the first one).
        - The second zero [0]: This is the class label. In this particular model, this field is reserved and not being used, so we use 0.
        - The variable i: This is the number of the discovered face. The model might find 5 faces, so i iterates through them one by one (face 1, 2, 3...).
        - The last number (and the most important): This specifies what information you want about this face. For each discovered face, the model returns 7 numbers (information) consecutively.
        """
        """
        - 0 ==> Batch ID
        - 1 ==> Class ID
        - 2 ==> Confidence
        - 3 ==> Start X (xmin)
        - 4 ==> Start Y (ymin)
        - 5 ==> End X (xmax)
        - 6 ==> End Y (ymax)
        """
        confidence = detections[0, 0, i, 2]
        # If confidence is low → ignore the face.
        if confidence < 0.5:
            continue
        
        
        """
        The output of the Caffe face detector returns the coordinates as follows: (x1_norm, y1_norm, x2_norm, y2_norm)
        If the model gives: (0.2, 0.1, 0.7, 0.6)
        
        x1 = 20% of the image width
        y1 = 10% of the image height
        x2 = 70% of the image width
        y2 = 60% of the image height
        But these are not pixels.
        
        So, We need to convert it to pixels
        """
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        # Converting coordinates to integers, Because the resulting coordinates are float
        x1, y1, x2, y2 = box.astype("int")
        
        # Protection against coordinates going outside the frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # We cut out the face from the original image in order to send it to the masked model.
        face = frame[y1:y2, x1:x2]
        # This is an additional protection; if, after all these calculations, the square comes out empty (its width is 0 or its length is 0), we ignore it so that the program doesn't crash.
        if face.size == 0:
            continue
        
        
        # Preparing the face for the mask model
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)    # Convert from BGR → RGB    
        face_resized = cv2.resize(face_rgb, (224, 224))     # Resize to 224x224
        face_array = img_to_array(face_resized) / 255.0     # Convert image to matrix and Normalization
        face_input = np.expand_dims(face_array, axis=0)     # Add an extra dimension so the image becomes batch size = 1
        
        # Make prediction, Value from 0 → 1
        pred = model.predict(face_input, verbose=0)[0][0]
        
        if pred > threshold:
            label = "Without Mask"
            color = (255, 0, 0)
            conf = pred
        else:
            label = "With Mask"
            color = (0, 255, 0)
            conf = 1 - pred
        
        # Save face result to (results) list 
        results.append({
            'box': (x1, y1, x2 - x1, y2 - y1),   # Converting coordinates from (x1, x2) to: (x, y, width, height)
            'label': label,
            'confidence': float(conf)
        })
    
    return results