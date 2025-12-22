import streamlit as st
import cv2
import time
from utils.camera import get_live_frame
from utils.detection import load_mask_model, detect_mask_dnn
from imutils.video import VideoStream
import numpy as np


def iou(box1, box2):
    x1, y1, w1, h1 = box1
    X1, Y1, W1, H1 = x1, y1, x1 + w1, y1 + h1
    
    x2, y2, w2, h2 = box2
    X2, Y2, W2, H2 = x2, y2, x2 + w2, y2 + h2
    
    xi1 = max(X1, X2)
    yi1 = max(Y1, Y2)
    xi2 = min(W1, W2)
    yi2 = min(H1, H2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area == 0:
        return 0.0
    
    box1_area = (W1 - X1) * (H1 - Y1)
    box2_area = (W2 - X2) * (H2 - Y2)
    
    union = box1_area + box2_area - inter_area
    return inter_area / union

# It places a large title at the top of the Streamlit page.
st.title("🎥 Live Mask Detection")

ip_url = st.session_state.get("ip_url", "")
threshold = st.session_state.get("threshold", 0.5)


# Make sure there is a camera link.
# If no camera is connected:
# A warning appears.
# `st.stop()` immediately stops the rest of the code.
if not ip_url:
    st.warning("Please go to the Settings page and enter the camera link.")
    st.stop()

# It shows the user which camera they are currently using.
st.write(f"Current Camera: {ip_url}")

# Loading the model
model = load_mask_model()

# Playing VideoStream
# Initialize the video stream from the IP address and start the background thread
# start(): This is the command that gives the camera a signal: "Start shooting and record the images to memory immediately."
vs = VideoStream(src=ip_url).start()
# Allow the camera sensor to warm up (auto-exposure adjustment)
time.sleep(1.0)

# empty(): function in Streamlit creates a placeholder container that can dynamically hold and update elements. This is particularly useful for replacing or clearing content in real-time without reloading the entire app.
frame_slot = st.empty()

# --------------------------------
# Creating Tracking (ID) variables
# --------------------------------
# List of people being followed
# If this is your first time opening the page, an empty list is created: Each item in it → Person (ID + bounding box, Last seen time)
if "tracked_people" not in st.session_state:
    st.session_state["tracked_people"] = []

# NEXT ID
# This is the ID counter:
# First person to enter → gets ID = 1
# Next → 2, then 3…
if "next_id" not in st.session_state:
    st.session_state["next_id"] = 1

# LIVE STATE
# It controls the on/off operation of the live stream.
# This is important because Streamlit replays the code with every interaction, so a constant state is necessary.
if "live" not in st.session_state:
    st.session_state["live"] = False

# ---------------------
# Live on/off functions
# ---------------------
# Live streaming
def start_live():
    st.session_state["live"] = True
    
# Stop the live stream
def stop_live():
    st.session_state["live"] = False

# ---------------
# Control buttons
# ---------------
# Start Button
# When pressed → Activates start_live() → Set "live" = True
st.button("Start Live Detection", on_click=start_live)

# Stop Button
# Pressing → makes "live" = False
st.button("Stop", on_click=stop_live)

# If the user presses "Start Live Detection" → live = True
if st.session_state["live"]:
    
    # Then the loop enters. The loop continues as long as `live` equals `True`. 
    while st.session_state["live"]:
        
        # get_live_frame() returns the last image coming from the VideoStream.
        frame = get_live_frame(vs)
        
        # If the camera loses connection → break.
        if frame is None:
            st.error("Lost connection to camera.")
            break
        
        # Running the model on the frame
        # The list returns the following format:
        """results.append({
            'box': (x1, y1, x2 - x1, y2 - y1),
            'label': label,
            'confidence': float(conf)
        })"""
        results = detect_mask_dnn(model, frame, threshold)
        
        # List of new people only (to register later). If a new person appears (new ID) → it will be placed in new_people.
        new_people = []
        
        # ------------
        # IOU Tracking
        # ------------
        # Rotate on each result detected in the current timeframe
        for r in results:
            box = r["box"]
            assigned_id = None
            
            # --------------
            # Tracking Logic
            # --------------
            # Attempting to match the current box with previously tracked individuals
            # The first time someone appears on camera ==> It will not be implemented on any element, because it simply is ==> for person in []: (empty)
            for person in st.session_state["tracked_people"]:
                # Calculate the intersection ratio (IoU) to confirm whether it is the same person or not
                # If IOU > 0.35 ==> This is the same person
                if iou(box, person["box"]) > 0.35:
                    assigned_id = person["id"]     # Recovering the old ID
                    person["box"] = box            # Update the box's location with the new coordinates
                    break
            
            # ------------------------------------
            # Registering a new person (New Entry)
            # ------------------------------------
            # If no match is found (assigned_id is still None)
            # If it's not there → Give it a new ID
            if assigned_id is None:
                assigned_id = st.session_state["next_id"]     # Pull a new ID
                st.session_state["next_id"] += 1              # Increase the counter for the next person
                
                # Add the person to the tracked_people list
                st.session_state["tracked_people"].append({
                    "id": assigned_id,
                    "box": box
                })
                
                # Add it to the "new_people" list so we only count it once in the statistics
                new_people.append(r)
            
            # Link the ID to the result to display it later
            # So that we can use it while drawing on the frame.
            r["id"] = assigned_id
        
        
        # --------------------------------------------
        # Draw the boxes and write the ID on the frame
        # --------------------------------------------
        for r in results:
            # Retrieve the person's data.
            x, y, w, h = r["box"]
            label = r["label"]
            conf = r["confidence"]
            person_id = r["id"]             # Person's ID
            
            # Color identification: Green for the mask, red without the mask
            color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
            
            # Box drawing
            # frame ==> The current image coming from the camera.
            # (x, y) ==> The top left dot of the box.
            # (x + w, y + h) ==> The bottom right point of the box.
            # color ==> Color of the box.
            # 2 ==> Text thickness..
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Write the text over the box (ID - Status - Confirmation percentage)
            cv2.putText(
                frame,                                          # The current image coming from the camera.
                f"ID {person_id} - {label} {conf*100:.1f}%",    # This is the text that will be displayed.
                (x, y - 10),                                    # Writing location: 10 pixels above the box.
                cv2.FONT_HERSHEY_SIMPLEX,                       # Font type.
                0.6,                                            # Text size.
                color,                                          # Text color (same color as the rectangle).
                2                                               # Text thickness.
            )
        
        # Display the final image in the Streamlit interface (with color conversion to RGB)
        frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        
        # --------------------------
        # Updating Records (Logging)        
        # --------------------------
        # If it's the first time we record logs → we create a new list.
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        
        """
        st.session_state["logs"]
        It is simply a variable stored within Streamlit, its function is to store data over time even if the code is rerun 1000 times.
        Streamlit reloads the entire page on every button click or update, 
        so we need to use session_state to store persistent values that won't be lost.
        """
        
        
        
        # Add a new record
        # How many new people appeared wearing a mask?
        # How many new people appeared without a mask?
        st.session_state["logs"].append({
            # new_people ==> This is a list of people who made their first appearance in the current frame.
            
            # Counting the number of people who appeared for the first time and "With Mask" for the current frame (This counts how many times this condition is true)
            "with_mask": sum(r["label"] == "With Mask" for r in new_people),
            # Counting the number of people who appeared for the first time and "Without Mask" for the current frame (This counts how many times this condition is true)
            "without_mask": sum(r["label"] == "Without Mask" for r in new_people)
        })
        """
        If you have two new values:
        First: With Mask → True
        Second: Without Mask → False
        The sum function only adds the True values. In Python, True = 1 and False = 0.
        Therefore, the result = 1
        """
        # The RESULTS WILL BE ==> 
        """
        logs = [
        {"with_mask": 1, "without_mask": 0}, # Frame showing a new person wearing a mask
        {"with_mask": 0, "without_mask": 2}, # Frame showing two new people without masks
        {"with_mask": 0, "without_mask": 0}, # Frame with no new people
        ...
        ]
        """
        # Slight delay to reduce processor load (Optional)
        time.sleep(0.05)
