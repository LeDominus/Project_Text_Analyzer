import torch
import logging
from models.model_manager import ModelManager

logger = logging.getLogger(__name__)

class StyleClassification:
    def __init__(self, model_manager: ModelManager):
        self.tokenizer_style = model_manager.tokenizer_style
        self.model_style = model_manager.model_style

    def classify_style(self, text: str) -> str:
        styles = [
            'Официально-деловой стиль',
            'Художественный стиль',
            'Научный стиль',
            'Публицистический стиль',
            'Разговорный стиль'
        ]

        if not text.strip():
            return "Пустой текст"

        try:
            inputs = self.tokenizer_style(
                text,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=256
            )
            with torch.no_grad():
                outputs = self.model_style(**inputs)
                
                logits = outputs.last_hidden_state[:, 0, :]
                
                predicted_class = torch.argmax(logits, dim=1).item()
                return styles[predicted_class] if predicted_class < len(styles) else "Не удалось определить стиль текста"

        except Exception as e:
            logger.error(f"Ошибка классификации стиля: {e}")
            return "Ошибка определения стиля"
