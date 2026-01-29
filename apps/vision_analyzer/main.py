from fastapi import FastAPI, UploadFile, File
import shutil
import os
from services.vision_analyzer import VisionAnalyzer

app = FastAPI()
analyzer = VisionAnalyzer()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections = analyzer.analyze(tmp_path)
    os.remove(tmp_path)

    return {"detections": detections}
