import streamlit as st

# Page Configuration
# Sets the browser tab title, favicon, and enables "Wide Mode" for better dashboard layout.
# NOTE: This command must be the very first Streamlit command in your script.
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide"       # It makes the page take up the entire width of the screen (very useful for the dashboard).
)

# Main Headers
st.title("Face Mask Detection App")
st.subheader('Choose from the side pages to start the project')

# 3. Project Documentation (Markdown)
# Using st.markdown allows writing formatted text (Headers, Bold, Lists) to explain the project scope.
st.markdown("""
## **1. Introduction 💡** 

### A. Overview

In the wake of recent global health challenges, particularly the COVID-19 pandemic, wearing face masks has become an essential and mandatory preventative measure in public and enclosed spaces. Governments and health organizations have enforced these policies to curb the transmission of airborne viruses.

### B. The Problem

Enforcing these policies and effectively monitoring compliance with mask-wearing protocols presents a significant challenge, especially in high-traffic areas such as shopping malls, airports, and public transportation hubs. Continuous manual monitoring is costly, labor-intensive, and prone to human error.

### C. The Proposed Solution

This is where Artificial Intelligence, specifically **Computer Vision**, plays a critical role. By developing an automated system, we can leverage existing surveillance cameras to analyze video streams or images in real-time to identify individuals who are not complying with mask-wearing guidelines.

### D. Project Objective

**The primary objective of this Notebook** is to build and train a deep learning model capable of:
1.  **Detecting** human faces within an image or video frame (Face Detection).
2.  **Classifying** each detected face accurately into one of two categories: **"With Mask"** or **"Without Mask"**.

""")

# streamlit run app.py
