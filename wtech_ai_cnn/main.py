from fastapi import FastAPI, Request, File, UploadFile 
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = FastAPI() #uygulamayı başlatır

model = load_model(r'D:\wtech\wtech_ai_cnn\images_cnn\models\adidas-nike.h5')

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(open("D:\wtech\wtech_ai_cnn\index.html", "r").read())

@app.get("/home") 
def home():
    return {"message": "CNN Modeli ile Ayakkabı Markası Tahmini Projesi"}

@app.get("/about") # /about endpointi için GET metodu
def about():
    return {"message": "CNN Modeli ve FastAPI geliştirmesi, Wtech ve IDSA ortaklığında yürütülen PYTHON ve Yapay Zeka eğitimi içerisinde yapılan bir çalışmadır."}

@app.get("/cnn", response_class=HTMLResponse)
async def cnn(request: Request):
    return HTMLResponse(open("D:\wtech\wtech_ai_cnn\cnn.html", "r").read())

@app.post("/guess_images/")
async def guess_images(file: UploadFile = File(...)):
    with open("temp_image.jpg", "wb") as f:
        f.write(await file.read())

    # Resmi yükle ve işle
    test_image = Image.open("temp_image.jpg")
    test_image = test_image.resize((240, 240))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0) / 255.0
    
    # Modelle tahmin yap
    result = model.predict(test_image)
    
    # Tahmin sonucuna göre sınıf etiketini belirle
    if result[0][0] > 0.5:
        guess = "Adidas"
    else:
        guess = "Nike"

    # Sonucu JSON olarak döndür
    return JSONResponse(content={"final": guess})



