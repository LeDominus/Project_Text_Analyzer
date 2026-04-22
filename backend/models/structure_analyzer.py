import re
import logging
from config import SECTION_PATTERNS, REFERENCE_STRUCTURE, SYNONYM_GROUPS

logger = logging.getLogger(__name__)

class StructureAnalyzer:
    def __init__(self) -> None:
        self.section_patterns = SECTION_PATTERNS
        self.reference_structure = REFERENCE_STRUCTURE
        self.synonim_groups = SYNONYM_GROUPS

    def extract_structure(self, text: str) -> set:
        structure = set()
        for pattern in self.section_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    section_title = match[0].lower().strip()
                else:
                    section_title = match.lower().strip()
                structure.add(section_title)
        return structure

    def _normalize_section_name(self, section: str) -> str:
        section_lower = section.lower()
        for canonical, variants in self.synonim_groups.items():
            if any(variant in section_lower for variant in variants):
                return canonical
        return section_lower

    def analyze_structure(self, text: str) -> tuple:
        raw_structure = self.extract_structure(text)
        normalized_structure = {self._normalize_section_name(sec) for sec in raw_structure}

        canonical_keys = set(self.synonim_groups.keys()) | {
            k for k in self.reference_structure.keys() 
            if not any(k in variants for variants in self.synonim_groups.values())
        }

        total_weight = 0.0
        matched_weight = 0.0
        missing_sections = []

        for key in canonical_keys:
            weight = self.reference_structure.get(key, 1.0)
            if key in self.synonim_groups:
                variants = self.synonim_groups[key]
                if any(var in normalized_structure for var in variants):
                    matched_weight += weight
                else:
                    missing_sections.append(key)
            else:
                if key in normalized_structure:
                    matched_weight += weight
                else:
                    missing_sections.append(key)
            total_weight += weight

        if total_weight == 0:
            logger.error("Эталонная структура имеет нулевой вес")
            return 0.0, "Ошибка: не задан эталон структуры"

        score = matched_weight / total_weight

        if score >= 0.75:
            interpretation = 'Структура текста соответствует стандартам УММ'
        elif 0.5 <= score < 0.75:
            interpretation = 'Структура текста требует доработки'
            if missing_sections:
                interpretation += f". Отсутствуют разделы: {', '.join(missing_sections[:3])}"
        else:
            interpretation = 'Структура текста не соответствует стандартам УММ'
            if missing_sections:
                interpretation += f". Обязательно добавить: {', '.join(missing_sections)}"

        return score, interpretation


