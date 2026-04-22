import asyncio
import torch
import logging
from models.text_preprocess import TextPreprocessor

logger = logging.getLogger(__name__)

class CoherenceAnalyzer:
    def __init__(self, embedding_model):
        self.model = embedding_model
        self.preprocessor = TextPreprocessor()

    async def coherence_analyze(self, text: str) -> dict:
        sections = self.preprocessor.split_into_sections(text)

        if len(sections) < 2:
            logger.warning("Недостаточно секций для анализа когерентности")
            return {
                'avg_coherence': 0.0,
                'interpretation': 'Недостаточно данных для анализа',
                'problem_zones': []
            }

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
            return {
                'avg_coherence': 0.0,
                'interpretation': 'Ошибка в данных',
                'problem_zones': []
            }

        avg_coherence = sum(coherence_scores) / len(coherence_scores)

        if avg_coherence >= 0.75:
            interpretation = 'Текст имеет связную структуру'
        elif 0.5 < avg_coherence < 0.75:
            interpretation = 'Текст имеет проблемы с логикой'
        else:
            interpretation = 'Текст логически не связан'

        problem_zones = []
        for i, score in enumerate(coherence_scores):
            if score < 0.5:
                problem_zones.append({
                    'section_from': i + 1,
                    'section_to': i + 2,
                    'similarity': round(score, 3)
                })

        return {
            'avg_coherence': avg_coherence,
            'interpretation': interpretation,
            'problem_zones': problem_zones
        }


