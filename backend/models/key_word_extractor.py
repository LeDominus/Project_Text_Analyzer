import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import asyncio
import re

class KeywordExtractor:
    def __init__(self, top_n=10):
        nltk.download('stopwords', quiet=True)
        russian_stopwords = stopwords.words('russian')
        self.vectorizer = TfidfVectorizer(
            stop_words=russian_stopwords,
            ngram_range=(1, 2),
            max_features=100
        )
        self.top_n = top_n

        self.blacklist = {"млн", "тыс", "руб", "долл", "кг", "л", "см", "м", "чел", "мес"}

    async def extract_keywords(self, text: str) -> list:
        def sync_tfidf():
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = np.array(self.vectorizer.get_feature_names_out())
            scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
            sorted_indices = np.argsort(scores)[::-1]

            keywords = []
            seen_words = set()

            for idx in sorted_indices:
                phrase = feature_names[idx].lower()

                if re.search(r'\d', phrase):
                    continue

                words = phrase.split()

                if any(word in self.blacklist for word in words):
                    continue

                if any(word in seen_words for word in words):
                    continue
                
                if any(len(word) <= 3 for word in words):
                    continue

                seen_words.update(words)
                keywords.append(phrase)

                if len(keywords) >= self.top_n:
                    break

            return keywords
        
        return await asyncio.to_thread(sync_tfidf)

