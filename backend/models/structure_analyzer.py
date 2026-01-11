import re
import logging

logger = logging.getLogger(__name__)

class StructureAnalyzer:
    section_patterns = [
        r'\b(введение|предисловие)\b',
        r'\b(глава|лекция|тема)\s*\d+(?:\.\d+)*\b',
        r'\bзаключение|вывод|итоги\b',
        r'\b(список использованной литературы|список рекомендованной литературы)\b',
        r'\bсодержание|оглавление\b',
        r'\bприложение\s*\d*\b',
        r'\bаннотация\b',
        r'\bсписок таблиц\b',
        r'\bсписок рисунков\b'
    ]

    @classmethod
    def extract_structure(cls, text: str) -> list:
        structure = []
        for pattern in cls.section_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    structure.append(match[0])
                else:
                    structure.append(match)
        return structure

    def analyze_structure(self, original_text: str, reference_text: str) -> tuple:
        """Анализ структуры текста"""
        original_structure = self.extract_structure(original_text)
        reference_structure = self.extract_structure(reference_text)

        if not reference_structure:
            logger.warning("Отсутствуют разделы в эталонном тексте")
            return 0.0, "Ошибка: нет эталонной структуры"

        weight_map = {
            'введение': 1.5,
            'заключение': 1.5,
            'глава': 1.0,
            'список использованной литературы': 1.0,
            'содержание': 2.0,
            'приложение': 1.0
        }

        matching_sections = set(original_structure) & set(reference_structure)

        structure_similarity = len(matching_sections) / len(reference_structure) if reference_structure else 0.0

        weighted_match = sum(weight_map.get(section.lower(), 1.0) for section in matching_sections)
        total_weight = sum(weight_map.get(section.lower(), 1.0) for section in reference_structure)
        weighted_similarity = weighted_match / total_weight if total_weight > 0 else 0.0

        weighted_result = (structure_similarity + weighted_similarity) / 2.0

        if weighted_result >= 0.75:
            interpretation = 'Структура текста соответствует стандартам'
        elif 0.5 < weighted_result < 0.75:
            interpretation = 'Структура текста требует доработки'
        else:
            interpretation = 'Структура текста не соответствует стандартам'

        return weighted_result, interpretation


