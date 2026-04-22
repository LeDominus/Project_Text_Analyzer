import nltk
import pymorphy3
import numpy as np
import asyncio
import re
from functools import lru_cache
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from config import BLACKLIST

class KeywordExtractor:
    def __init__(self, top_n=10):
        nltk.download('stopwords', quiet=True)
        
        self.morph = pymorphy3.MorphAnalyzer()
        russian_stopwords = stopwords.words('russian')
        self.top_n = top_n
        self.blacklist = BLACKLIST
        
        self.vectorizer = TfidfVectorizer(
            stop_words=russian_stopwords,
            ngram_range=(1, 2),
            max_features=100
        )

    @lru_cache(maxsize=256)
    def _lemmatize_word(self, word: str) -> str:
        """Лемматизация отдельного слова с кэшированием"""
        return self.morph.parse(word)[0].normal_form

    def _lemmatize_phrase(self, phrase: str) -> str:
        """Лемматизация всей фразы (для n-грамм)"""
        words = phrase.split()
        lemmas = [self._lemmatize_word(w) for w in words]
        return " ".join(lemmas)

    async def extract_keywords(self, text: str) -> list:
        @lru_cache(maxsize=64)
        def sync_tfidf():
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = np.array(self.vectorizer.get_feature_names_out())
            scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
            sorted_indices = np.argsort(scores)[::-1]

            keywords = []
            seen_lemmas = set()
            seen_phrases = set()

            for idx in sorted_indices:
                phrase = feature_names[idx].lower()

                if re.search(r'\d', phrase):
                    continue

                words = phrase.split()
                
                if any(word in self.blacklist for word in words):
                    continue

                if any(len(word) <= 2 for word in words):
                    continue

                lemma_phrase = self._lemmatize_phrase(phrase)
                
                if lemma_phrase in seen_phrases:
                    continue

                lemmas_set = set(lemma_phrase.split())
                if lemmas_set & seen_lemmas:
                    continue

                seen_lemmas.update(lemmas_set)
                seen_phrases.add(lemma_phrase)
                keywords.append(phrase)

                if len(keywords) >= self.top_n:
                    break

            return keywords
        
        return await asyncio.to_thread(sync_tfidf)

