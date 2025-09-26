import asyncio
import torch
import logging
from models.text_preprocess import TextPreprocessor

logger = logging.getLogger(__name__)

class CoherenceAnalyzer:
    def __init__(self, embedding_model):
        self.model = embedding_model  # SentenceTransformer
        self.preprocessor = TextPreprocessor()

    async def coherence_analyze(self, text: str) -> tuple:
        sections = self.preprocessor.split_into_sections(text)

        if len(sections) < 2:
            logger.warning("Недостаточно секций для анализа когерентности")
            return 0.0, 'Недостаточно данных для анализа'

        # Асинхронно получаем эмбеддинги через to_thread, потому что encode синхронный
        embeddings = await asyncio.gather(*[
            asyncio.to_thread(self.model.encode, s, convert_to_tensor=True)
            for s in sections
        ])

        coherence_scores = []
        for i in range(1, len(embeddings)):
            similarity = torch.nn.functional.cosine_similarity(
                embeddings[i-1],
                embeddings[i],
                dim=-1
            )
            coherence_scores.append(similarity.mean().item())

        if not coherence_scores:
            logger.error("Не удалось рассчитать когерентность")
            return 0.0, 'Ошибка в данных'

        avg_coherence = sum(coherence_scores) / len(coherence_scores)

        if avg_coherence > 0.85:
            interpretation = 'Текст имеет связную структуру'
        elif 0.5 < avg_coherence <= 0.85:
            interpretation = 'Текст имеет проблемы с логикой'
        else:
            interpretation = 'Текст логически не связан'

        return avg_coherence, interpretation


