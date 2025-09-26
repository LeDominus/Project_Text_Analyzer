import os
import logging
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas.responses import AnalysisResponse
from dotenv import load_dotenv

load_dotenv()

from models.text_analyzer import TextAnalyzer

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Поддерживаются только PDF файлы")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    reference_path = os.getenv('REFERENCE_PATH')
    logging.info(f"REFERENCE_PATH={reference_path}")
    
    if not reference_path:
        raise RuntimeError("Переменная окружения REFERENCE_PATH не задана")
    if not os.path.exists(reference_path):
        raise RuntimeError(f"Файл по пути {reference_path} не найден")


    try:
        analyzer = TextAnalyzer()
        results = await analyzer.analyze_document(original_path=file_path, reference_path=reference_path)
        return results

    except Exception as e:
        logging.error(f"Ошибка анализа: {str(e)}")
        logging.error(traceback.format_exc())  
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе: {str(e)}")


@app.get("/")
def main():
    return {"message": "OK"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5002, reload=True)


