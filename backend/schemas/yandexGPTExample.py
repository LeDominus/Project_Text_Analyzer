from openai import OpenAI
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

YANDEX_API_KEY = 'AQVNyjBM8YPCiKP5fiIoJkpV_kT4lXpXmzj1P1lp'
YANDEX_FOLDER = "b1gimp7lrghrvurlm2gq"
MODEL_NAME = "qwen3-235b-a22b-fp8"
MODEL_VERSION = "latest"

if not YANDEX_API_KEY or not YANDEX_FOLDER:
    logging.error("Не задан YANDEX_API_KEY или YANDEX_FOLDER")
    exit(1)

MODEL_URI = f"gpt://{YANDEX_FOLDER}/{MODEL_NAME}/{MODEL_VERSION}"

client = OpenAI(
    api_key=YANDEX_API_KEY,
    base_url="https://llm.api.cloud.yandex.net/v1"
)

text_to_analyze = "Привет! Проанализируй этот короткий текст учебного материала."

try:
    response = client.chat.completions.create(
        model=MODEL_URI,
        messages=[
            {"role": "system", "content": "Ты эксперт по анализу учебно-методических материалов."},
            {"role": "user", "content": text_to_analyze}
        ],
        temperature=0.6,
        max_tokens=500,
    )

    logging.info(f"MODEL_URI: {MODEL_URI}")
    logging.info(f"Response object: {response}")
    logging.info(f"Ответ модели: {response.choices[0].message.content}")

    print("✅ Анализ текста:")
    print(response.choices[0].message.content)

except Exception as e:
    logging.error(f"Ошибка при вызове YandexGPT: {e}")



