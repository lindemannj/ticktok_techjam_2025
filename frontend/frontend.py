import base64
import streamlit as st
import requests

# Set a constant for the API URL for maintainability
API_URL = "http://tiktok-tech-jam-api-gateway:8000"

# Initialize session state variables to manage app flow
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

st.title("Image Anonymization App")
st.subheader("Upload an image to detect and optionally blur faces.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

    with st.spinner('Processing...'):
        try:

            response = requests.post(f"{API_URL}/process-image/",
                                     files={"file": uploaded_file.getvalue()})
            response.raise_for_status()
            processed_image_string = response.json()["processed_image_string"]
            processed_image_bytes = base64.b64decode(processed_image_string)

            # Display the comparison of original and processed images
            st.header("Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.uploaded_file, caption="Original Image")
            with col2:
                st.image(processed_image_bytes, caption="Processed Image")

            st.download_button(
                label="Download Processed Image",
                data=processed_image_bytes,
                file_name="processed_image.jpeg",
                mime="image/jpeg"
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Error during blurring API call: {e}")
