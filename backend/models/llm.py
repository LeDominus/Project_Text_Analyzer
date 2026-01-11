import os
import json
import logging
import warnings
from dotenv import load_dotenv
from openai import OpenAI
from schemas.prompt import PROMPT

load_dotenv()
warnings.filterwarnings('ignore')

YANDEX_FOLDER = os.getenv('YANDEX_FOLDER')
MODEL_NAME = os.getenv('MODEL_NAME')
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')

TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 500

class LLMApi:
    def __init__(self):
        self.api_key = YANDEX_API_KEY
        self.folder_id = YANDEX_FOLDER
        self.model_uri = f"gpt://{self.folder_id}/{MODEL_NAME}"
        
        if not self.api_key:
            raise ValueError("Не удалось найти YANDEX_API_KEY")
        if not self.folder_id:
            raise ValueError("Не удалось найти YANDEX_FOLDER")

        self.client = OpenAI(
            api_key=os.getenv("YANDEX_API_KEY"),
            base_url="https://rest-assistant.api.cloud.yandex.net/v1",
            default_headers={
                "X-Project-Id": os.getenv("YANDEX_FOLDER")
            }
        )

    def get_common_recommendation(
        self,
        text_material: str,
        analysis_results: dict,
    ) -> str:
        """
        Получить одну общую рекомендацию на основе всех данных анализа
        """
        
        user_content = PROMPT.format(
            text_material=text_material,
            analysis_results=json.dumps(
                analysis_results,
                ensure_ascii=False,
                indent=2
            )
        )

        try:
            response = self.client.responses.create(
                model=f"gpt://{os.getenv('YANDEX_FOLDER')}/{os.getenv('YANDEX_MODEL')}",
                input=user_content,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE
            )

            logging.info(f"RAW response from Yandex: {response}")
            return response.output[0].content[0].text

        except Exception as e:
            logging.error(f"Ошибка при вызове YandexGPT: {e}", exc_info=True)
            return "Не удалось сформировать рекомендацию. Пожалуйста, проверьте анализ по отдельным критериям"

