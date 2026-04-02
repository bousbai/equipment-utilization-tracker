import streamlit as st
import cv2
import numpy as np

# Setting title of the web application
st.title("Equipment Utilization Tracker Dashboard")

# Initialize video capture
video_source = "path_to_your_video_source"  # You may replace this with a video file or camera source
cap = cv2.VideoCapture(video_source)

# Define a function to display video feed with bounding boxes
def show_video():
    frame_window = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("No video feed available.")
            break
        
        # Simulate bounding boxes and activity classification
        bounding_boxes = [(50, 50, 200, 200)]  # Example bounding box coordinates
        activity_classification = "Activity: Working"  # Placeholder activity classification

        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Convert the BGR image to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Show video frame
        frame_window.image(frame, channels="RGB")
        
        # Here you can break the loop to avoid an infinite loop in demo
        if st.button('Stop'):
            break

# Equipment status cards (placeholder data)
st.subheader("Equipment Status")
equipment_status = {
    "Equipment 1": "Operational",
    "Equipment 2": "Under Maintenance",
    "Equipment 3": "Operational"
}
for equipment, status in equipment_status.items():
    st.card(equipment, status)

# Real-time utilization metrics (placeholder data)
st.subheader("Real-time Utilization Metrics")
st.write("Utilization Rate: 75%")
st.write("Active Equipment Count: 3")

# Start showing the video feed
show_video()

# Cleanup on exit
cap.release()