import asyncio
from .model_manager import ModelManager
from .structure_analyzer import StructureAnalyzer
from .coherence_analyzer import CoherenceAnalyzer
from .readability_analyzer import ReadabilityAnalyzer
from models.key_word_extractor import KeywordExtractor
from models.text_preprocess import TextPreprocessor
from models.responses import AnalysisResult


class TextAnalyzer:
    def __init__(self):
        self.model_manager = ModelManager()
        self.preprocessor = TextPreprocessor()
        self.structure_analyzer = StructureAnalyzer()
        self.coherence_analyzer = CoherenceAnalyzer(self.model_manager.model_bert)
        self.readability_analyzer = ReadabilityAnalyzer()
        self.keyword_extractor = KeywordExtractor()

    def _conv_to_perc(self, result):
        """Конвертация результатов в проценты"""
        if isinstance(result, (int, float)) and 0 <= result <= 1:
            return result * 100
        return result

    async def analyze_document(self, original_path: str):
        try:
            original_text = self.preprocessor.extract_text_from_pdf(original_path)
            
            coherence_result = await self.coherence_analyzer.coherence_analyze(original_text)
            structure_result = await asyncio.to_thread(self.structure_analyzer.analyze_structure, original_text)
            read_result = await self.readability_analyzer.analyze_readability(original_text)
            keywords = await self.keyword_extractor.extract_keywords(original_text)

            return AnalysisResult(
                coherence_result=self._conv_to_perc(coherence_result['avg_coherence']),
                coherence_interpretation=coherence_result['interpretation'],
                coherence_problem_zones=coherence_result['problem_zones'],
                structure_result=self._conv_to_perc(structure_result[0]),
                structure_interpret=structure_result[1],
                read_result=read_result,
                keywords=keywords
            )

        except Exception as e:
            raise ValueError(f"Ошибка при анализе: {e}")