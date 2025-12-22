import streamlit as st
import pandas as pd

# Set the dashboard title
st.title('📊 Dashboard')

# Validation: Check if 'logs' exist in session_state and are not empty
# This prevents crashes if the dashboard is accessed before detection starts
if 'logs' not in st.session_state or len(st.session_state['logs']) == 0:
    st.info("No data yet. Take a snapshot first.")
else:
    # Convert the list of log dictionaries into a Pandas DataFrame for easier plotting
    df = pd.DataFrame(st.session_state['logs'])
    
    # 1. Show Raw Data Table
    st.subheader('Statistics:')
    st.dataframe(df)
    
    # 2. Visualize Trends (Line Charts)
    # Plotting 'Without Mask' counts over time (snapshots)
    st.subheader('Number of people without masks throughout history:')
    st.line_chart(df['without_mask'])
    
    # Plotting 'With Mask' counts over time
    st.subheader('Number of people with masks throughout history:')
    st.line_chart(df['with_mask'])