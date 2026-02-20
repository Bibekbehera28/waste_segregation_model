from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import io
from PIL import Image

model = tf.keras.models.load_model("best_waste_model.keras")

# ⚠ MUST MATCH train_generator.class_indices ORDER
class_labels = [
    "Electronic waste",
    "Hazardous",
    "Non-Recyclable",
    "Organic",
    "Recyclable"
]

app = FastAPI()

def predict_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    return class_labels[predicted_class], confidence

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, or PNG supported.")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    label, confidence = predict_image(img)

    return {
        "filename": file.filename,
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }