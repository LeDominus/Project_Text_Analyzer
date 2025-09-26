import nltk
import numpy as np
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

class KeywordExtractor:
    def __init__(self, top_n=10):
        nltk.download('stopwords')
        russian_stopwords = stopwords.words('russian')
        self.vectorizer = TfidfVectorizer(stop_words=russian_stopwords, ngram_range=(1, 2), max_features=20)
        self.top_n = top_n

    async def extract_keywords(self, text: str) -> list:
        def sync_tfidf():
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = np.array(self.vectorizer.get_feature_names_out())
            scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
            top_indices = np.argsort(scores)[::-1][:self.top_n]
            return feature_names[top_indices].tolist()
        
        return await asyncio.to_thread(sync_tfidf)
