import streamlit as st
import pandas as pd
from PIL import Image

# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Display the data using Streamlit
st.title("Handwriting recognition")
st.markdown("---")

# Logo and title
logo_image = Image.open("logo.png")
st.image(logo_image, width=200)
st.markdown("---")

# Trained Data
st.write("Trained Data")
train_df = train.sample(n=100)  # Sample 100 rows from the dataframe
st.line_chart(train_df)
st.markdown("---")

# Tested Data
st.write("Tested Data")
test_df = test.sample(n=100)  # Sample 100 rows from the dataframe
st.line_chart(test_df)

# Rest of your code...
# Add your test_prediction function and other functions here
