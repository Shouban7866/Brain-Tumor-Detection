import numpy as np
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import io

app = FastAPI(title="Brain Tumor Detection API")

MODEL_PATH = "cnn_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model load karne mein masala hua: {str(e)}")

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.get("/")
def home():
    return {"message": "Brain Tumor Detection API is Running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]

        all_probabilities = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}

        return {
            "status": "success",
            "prediction": predicted_class,
            "confidence": f"{confidence * 100:.2f}%",
            "all_probabilities": all_probabilities
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

