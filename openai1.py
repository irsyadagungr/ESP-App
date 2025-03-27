import streamlit as st

st.title("Three Columns")

# Create a blank space
st.markdown("")

# Create a left column
left_column = st.sidebar.empty()

# Add content to the left column
left_column.markdown("This is the left column.")
left_column.selectbox("Choose an option:", ["Option 1", "Option 2", "Option 3"])
left_column.slider("Select a value:", 0, 100, 50)

# Create a center column
center_column = st.empty()

# Add content to the center column
center_column.markdown("This is the center column.")
center_column.checkbox("Check this box")
center_column.radio("Select an option:", ["Option A", "Option B", "Option C"])

# Create a right column
right_column = st.sidebar.empty()

# Add content to the right column
right_column.markdown("This is the right column.")
right_column.text_input("Enter some text:")
right_column.button("Click me!")