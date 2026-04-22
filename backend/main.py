import os
import warnings
import logging
import traceback
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from models.llm import LLMApi
from models.responses import AnalysisResult
from models.text_analyzer import TextAnalyzer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logger.info("==== [INFO] Приложение запущено! ==== ")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Поддерживаются только PDF файлы")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    logger.info("==== [INFO] Анализируем... ==== ")
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        analyzer = TextAnalyzer()
        llm = LLMApi()

        analysis_results = await analyzer.analyze_document(original_path=file_path)

        common_recommendation = llm.get_common_recommendation(
            text_material=analysis_results.summary,
            analysis_results=analysis_results.dict()
        )

        results = AnalysisResult(
            **analysis_results.dict(exclude={"recommendation"}),
            recommendation=common_recommendation 
        )
        
        logger.info("==== [INFO] Анализ завершён успешно! ==== ")
        return results

    except HTTPException:
        raise

    except Exception as e:
        logging.error(f"Ошибка анализа: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Ошибка при анализе документа"
        )

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/")
def main():
    return {"message": "OK"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5002, reload=True)
