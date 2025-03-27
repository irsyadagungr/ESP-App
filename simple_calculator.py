import streamlit as st

# Title of the App
st.title("Simple Math Calculator")

# Input fields
num1 = st.number_input("Enter first number", value=0.0, format="%.6f")
num2 = st.number_input("Enter second number", value=0.0, format="%.6f")

# Dropdown for selecting operation
operation = st.selectbox("Select Operation", ["+", "-", "×", "÷"])

# Calculate result
result = None
if operation == "+":
    result = num1 + num2
elif operation == "-":
    result = num1 - num2
elif operation == "×":
    result = num1 * num2
elif operation == "÷":
    result = num1 / num2 if num2 != 0 else "Error (Division by Zero)"

# Display the result
st.write("### = ", result)

import os
import tensorflow as tf

# Get the absolute path of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "my_model.h5")

# Load the model
if os.path.exists(model_path):
    new_model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully in Streamlit!")
else:
    raise FileNotFoundError(f"❌ Model file NOT found: {model_path}")
