import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="centered")

st.title("🧠 Brain Tumor Detection")
st.write("Please upload the MRI scan so that the ResNet50 model can detect the tumor.")

FASTAPI_URL = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("Please select your MRI image here (JPG/PNG)....", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    if st.button("Analyze Scan 🔍"):
        with st.spinner("The model is analyzing the image..."):
            try:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                img_byte_arr = img_byte_arr.getvalue()

                files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}

                response = requests.post(FASTAPI_URL, files=files)

                if response.status_code == 200:
                    result = response.json()

                    if result.get("status") == "success":
                        st.success("Analysis Complete!")

                        st.metric(label="Predicted Class", value=result["prediction"])
                        st.metric(label="Confidence Level", value=result["confidence"])

                        st.subheader("Class Probabilities")
                        st.bar_chart(result["all_probabilities"])
                    else:
                        st.error(f"Error: {result.get('message')}")
                else:
                    st.error(f"Unable to connect to the backend API. Status code: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("The FastAPI server is not running. Please start the backend first.")
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")