import asyncio
from .model_manager import ModelManager
from .style_classification import StyleClassification
from .structure_analyzer import StructureAnalyzer
from .coherence_analyzer import CoherenceAnalyzer
from .readability_analyzer import ReadabilityAnalyzer
from models.key_word_extractor import KeywordExtractor
from models.text_preprocess import TextPreprocessor

import asyncio

class TextAnalyzer:
    def __init__(self):
        self.model_manager = ModelManager()
        self.preprocessor = TextPreprocessor()
        self.structure_analyzer = StructureAnalyzer()
        self.coherence_analyzer = CoherenceAnalyzer(self.model_manager.model_bert)
        self.readability_analyzer = ReadabilityAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        self.style_classifier = StyleClassification(self.model_manager)

    async def analyze_document(self, original_path: str, reference_path: str):
        try:

            original_text = self.preprocessor.extract_text_from_pdf(original_path)
            reference_text = self.preprocessor.extract_text_from_pdf(reference_path)

            style_result = self.style_classifier.classify_style(original_text)

            coherence_result = await self.coherence_analyzer.coherence_analyze(original_text)

            structure_result = await asyncio.to_thread(
                self.structure_analyzer.analyze_structure,
                original_text,
                reference_text
            )

            read_result = await self.readability_analyzer.analyze_readability(original_text)
            keywords = await self.keyword_extractor.extract_keywords(original_text)

            return {
                "style_result": style_result,
                "coherence_result": coherence_result[0],            
                "coherence_interpretation": coherence_result[1],   
                "structure_result": structure_result[0],
                "structure_interpret": structure_result[1],
                "read_result": read_result,                        
                "keywords": keywords
            }

        except Exception as e:
            raise ValueError(f"Ошибка при анализе: {e}")




    
    

