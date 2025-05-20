import asyncio
import logging
import re
import numpy as np
import torch
import pdfplumber
import textstat
import nltk

from transformers import (
    AutoTokenizer,
    AutoModel,
    BartModel,
    BertTokenizer,
    BartTokenizer,
    BertForSequenceClassification
)
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from tqdm.asyncio import tqdm_asyncio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self):
        """Инициализация моделей и токенизаторов"""
        self._init_bert_models()
        self._init_bart_models()
        self._init_style_classifier()
        self._init_keyword_extractor()

    def _init_bert_models(self):
        """Инициализация BERT моделей"""
        self.tokenizer_bert = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny')
        self.model_bert = AutoModel.from_pretrained('cointegrated/rubert-tiny')

    def _init_bart_models(self):
        """Инициализация BART моделей"""
        self.tokenizer_bart = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
        self.model_bart = BartModel.from_pretrained('sshleifer/distilbart-cnn-12-6')

    def _init_style_classifier(self):
        """Инициализация классификатора стилей"""
        self.tokenizer_style = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model_style = BertForSequenceClassification.from_pretrained("any0019/text_style_classifier")

    def _init_keyword_extractor(self):
        """Инициализация модели для извлечения ключевых слов"""
        nltk.download('stopwords')
        russian_stopwords = stopwords.words('russian')
        self.vectorizer = TfidfVectorizer(stop_words=russian_stopwords, ngram_range=(1, 2), max_features=20)

    @staticmethod
    def convert_text_from_pdf(file_path: str) -> str:
        """Извлечение текста из PDF файла"""
        full_text = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text.append(text)
            return '\n'.join(full_text)
        except Exception as e:
            logger.error(f"Ошибка при чтении PDF: {e}")
            return ""

    async def get_embeddings(self, text: str) -> torch.Tensor:
        """Получение эмбеддингов текста"""
        inputs = self.tokenizer_bart(
            text, 
            return_tensors='pt', 
            max_length=512, 
            truncation=True, 
            padding='max_length'
        )
        with torch.no_grad():
            outputs = self.model_bart(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def classify_style(self, text: str) -> str:
        """Классификация стиля текста"""
        styles = [
            'Официально-деловой стиль',
            'Художественный стиль',
            'Научный стиль',
            'Публицистический стиль',
            'Разговорный стиль'
        ]
        
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
                
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            return styles[predicted_class] if predicted_class < len(styles) else "Неизвестный стиль"
        except Exception as e:
            logger.error(f"Ошибка классификации стиля: {e}")
            return "Ошибка определения стиля"

    @staticmethod
    def extract_structure(text: str) -> list:
        """Извлечение структуры текста"""
        section_patterns = [
            r'\b(введение|предисловие)\b',
            r'\b(глава|лекция|тема)\s*\d+(\.\d+)*\b',
            r'\bзаключение|вывод|итоги\b',
            r'\b(список использованной литературы|список рекомендованной литературы)\b',
            r'\bсодержание|оглавление\b',
            r'\bприложение\s*\d*\b',
            r'\bаннотация\b',
            r'\bсписок таблиц\b',
            r'\bсписок рисунков\b'
        ]
        
        structure = []
        for pattern in section_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                structure.append(matches[0])
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

        if weighted_result > 0.85:
            interpretation = 'Структура текста соответствует стандартам'
        elif 0.5 < weighted_result <= 0.85:
            interpretation = 'Структура текста требует доработки'
        else:
            interpretation = 'Структура текста не соответствует стандартам'

        return weighted_result, interpretation

    async def analyze_coherence(self, text: str) -> tuple:
        """Анализ когерентности текста"""
        lines = text.split('\n')
        sections = [' '.join(lines[i:i+30]) for i in range(0, len(lines), 30)]
        
        if len(sections) < 2:
            logger.warning("Недостаточно секций для анализа когерентности")
            return 0.0, 'Недостаточно данных для анализа'
        
        embeddings = await asyncio.gather(*[ 
            self.get_embeddings(section) for section in sections
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

    @staticmethod
    async def analyze_readability(text: str) -> dict:
        """Анализ читаемости текста"""
        async def _readability_level(score: float) -> str:
            if score > 90: return "Для младших классов"
            elif score > 60: return "Для средней школы"
            elif score > 30: return "Для студентов ВУЗов"
            else: return "Для специалистов"

        tasks = [
            asyncio.to_thread(textstat.flesch_reading_ease, text),
            asyncio.to_thread(textstat.gunning_fog, text),
            asyncio.to_thread(lambda: len(re.split(r'[.!?]+', text))),
            asyncio.to_thread(lambda: len(re.findall(r'\w+', text))),
            asyncio.gather(*[ 
                asyncio.to_thread(textstat.syllable_count, word) 
                for word in re.findall(r'\w+', text)
            ])
        ]

        flesch_score, gunning_score, sentences, words, syllables = await asyncio.gather(*tasks)
        
        return {
            "Индекс Флеша": flesch_score,
            "Индекс Ганнинга": gunning_score,
            "Количество предложений": sentences,
            "Количество слов": words,
            "Слоги": sum(syllables),
            "Сложность текста": await _readability_level(flesch_score)
        }

    async def extract_keywords(self, text: str, top_n: int = 10):
        try:
            def sync_tfidf():
                tfidf_matrix = self.vectorizer.fit_transform([text])
                feature_names = np.array(self.vectorizer.get_feature_names_out())
                scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
                top_indices = np.argsort(scores)[::-1][:top_n]
                return feature_names[top_indices].tolist()
            
            return await asyncio.to_thread(sync_tfidf)

        except Exception as e:
            logger.error(f"Ошибка при извлечении ключевых слов: {e}")
            return []

    async def _show_loading(self, task):
        """Отображение прогресса выполнения задачи"""
        progress_bar = tqdm_asyncio(total=1, desc='Обработка', unit='task')
        result = await task
        progress_bar.update(1)
        progress_bar.close()
        return result

    async def analyze_document(self, original_path: str, reference_path: str) -> dict:
        """Основной метод анализа документа"""
        original_text = self.convert_text_from_pdf(original_path)
        reference_text = self.convert_text_from_pdf(reference_path)

        if not original_text or not reference_text:
            raise ValueError("Не удалось прочитать один из документов")

        tasks = {
            "style_result": self.classify_style(original_text),
            "coherence_result": self.analyze_coherence(original_text),
            "structure_result": asyncio.to_thread(
                self.analyze_structure, 
                original_text, 
                reference_text
            ),
            "read_result": self.analyze_readability(original_text),
            "keywords_result": self.extract_keywords(original_text, top_n=10)
        }

        results = {}
        for name, task in tasks.items():
            if asyncio.iscoroutine(task):
                results[name] = await self._show_loading(task)
            else:
                results[name] = await self._show_loading(asyncio.to_thread(task))

        return results

async def main():
    analyzer = TextAnalyzer()
    try:
        result = await analyzer.analyze_document(
            original_path='C:\\Programming\\Project_BERT\\data\\PDF_учебники\\201_Stat.pdf',
            reference_path='C:\\Programming\\Project_BERT\\data\\PDF_учебники\\1_УчП_Эконометрика_РД_Воскобойников.pdf'
        )
        print(result)
    except Exception as e:
        logger.error(f"Ошибка при анализе: {e}")

if __name__ == '__main__':
    asyncio.run(main())


    
    

