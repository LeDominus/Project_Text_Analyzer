import asyncio
import re

class ReadabilityAnalyzer:
    async def analyze_readability(self, text: str) -> dict:
        """Анализ читаемости текста для русского языка"""
        
        async def _readability_level(score: float) -> str:
            """Интерпретация индекса Флеша"""
            if score > 90: return "Для младших классов"
            elif score > 60: return "Для средней школы"
            elif score > 30: return "Для студентов ВУЗов"
            else: return "Для специалистов"

 
        words = re.findall(r'\w+', text)
        sentences = re.split(r'[.!?]+', text)

        tasks = [
            asyncio.to_thread(lambda: len(sentences)),
            asyncio.to_thread(lambda: len(words)),
            asyncio.to_thread(lambda: sum(len(re.findall(r'[аеёиоуыэюя]', w, re.IGNORECASE)) for w in words))
        ]

        num_sentences, num_words, num_syllables = await asyncio.gather(*tasks)
        flesch_score = 206.835 - 1.3 * (num_words / max(num_sentences,1)) - 60 * (num_syllables / max(num_words,1))

        complexity = await _readability_level(flesch_score)

        return {
            "Индекс Флеша (русский)": round(flesch_score, 2),
            "Количество предложений": num_sentences,
            "Количество слов": num_words,
            "Количество слогов": num_syllables,
            "Сложность текста": complexity
        }



