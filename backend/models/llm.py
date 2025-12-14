import os
import json
import logging
from openai import OpenAI
from schemas.prompt import MODULE_PROMPT

class LLMApi:
    def __init__(self):
        YANDEX_FOLDER = "b1gimp7lrghrvurlm2gq"
        MODEL_NAME = "qwen3-235b-a22b-fp8"
        MODEL_VERSION = "latest"
        
        self.api_key = os.getenv("YANDEX_API_KEY")
        self.model_uri = f"gpt://{YANDEX_FOLDER}/{MODEL_NAME}/{MODEL_VERSION}"

        if not self.api_key:
            raise ValueError("Не удалось найти YANDEX_ID_KEY")
        if not self.model_uri:
            raise ValueError("Не удалось найти MODEL_URI")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://llm.api.cloud.yandex.net/v1"
        )

    def chat(self, text_material: str, analysis_results: dict, temperature=0.7, max_tokens=2000):
        user_content = MODULE_PROMPT.format(
            text_material=text_material,
            analysis_results=json.dumps(analysis_results, ensure_ascii=False, indent=2)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_uri,
                messages=[
                    {"role": "system", "content": "Ты — эксперт по анализу учебно-методических материалов."},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            logging.info(f"RAW response from YandexGPT: {response}")

            result_text = None
            if response.choices and response.choices[0].message:
                result_text = response.choices[0].message.content

            if not result_text:
                result_text = "Модель не вернула текст."

            return result_text

        except Exception as e:
            logging.error(f"Ошибка при вызове YandexGPT: {e}")
            return "Ошибка при вызове YandexGPT"


