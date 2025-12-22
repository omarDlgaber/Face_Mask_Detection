import streamlit as st

# Set the dashboard title
st.title("⚙️ Settings")

# Input for Camera URL
# We use .get() to pre-fill the box with the saved value if it exists, otherwise use the default IP
ip_url = st.text_input("IP Camera URL", st.session_state.get("ip_url", "http://192.168.1.12:8080/video"))

# Slider for Model Sensitivity (Threshold)
# Allows adjusting how strict the model is (Higher = fewer false positives)
threshold = st.slider("Mask Detection Threshold", 0.3, 0.9, st.session_state.get("threshold", 0.5))

# Save Button Logic
if st.button("Save Settings"):
    
    # Update the global session state so these values are accessible in other pages
    st.session_state["ip_url"] = ip_url
    st.session_state["threshold"] = threshold
    st.success('Settings saved')