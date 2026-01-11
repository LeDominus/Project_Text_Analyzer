

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


if __name__ == "__main__":
    client = OpenAI(
    api_key=os.getenv("YANDEX_API_KEY"),
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    default_headers={
        "X-Project-Id": os.getenv("YANDEX_FOLDER")
    }
    )

    response = client.responses.create(
        model="gpt://b1gimp7lrghrvurlm2gq/gemma-3-27b-it/latest",
        # model=f"gpt://{os.getenv('YANDEX_FOLDER')}/{os.getenv('YANDEX_MODEL')}",
        input="Объясни, что такое RAG простыми словами",
        max_output_tokens=300,
        temperature=0.3
    )

    import json
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
