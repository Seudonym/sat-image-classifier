from fastapi import FastAPI, UploadFile, File
import uvicorn

from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model('../models/1')
classes = ['Cloudy', 'Desert', 'Forest', 'Water']

@app.post('/api/predict')
async def predict(
    file: UploadFile = File(...)
):
    data = await file.read()
    data = Image.open(BytesIO(data))
    data = data.resize((128, 128))

    img = np.array(data)
    print(img)
    img = np.expand_dims(img, 0)
    
    prediction = model.predict(img)[0]

    confidence = 100 * np.max(prediction)
    predicted_class = classes[np.argmax(prediction)]

    return {
        'class': predicted_class,
        'confidence': confidence
    }
     

if __name__ == '__main__':
    uvicorn.run('main:app', host='localhost', port=8000, reload=True)