from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()
app.mount("/static", StaticFiles(directory="project/static"), name="static")
templates = Jinja2Templates(directory="project/templates")

model = tf.keras.models.load_model("model.h5")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).resize((128, 128)).convert("RGB")
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0])) * 100

        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        result = classes[predicted_class]
        result_text = f"{result} ({confidence:.2f}%)"

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": result_text
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"Error: {str(e)}"
        })
