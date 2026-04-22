import warnings
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, T5ForConditionalGeneration
warnings.filterwarnings('ignore')

class ModelManager:
    def __init__(self):
        self._init_bert()
        self._init_summarizer()

    def _init_bert(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_bert = SentenceTransformer("cointegrated/rubert-tiny", device = device)

    def _init_summarizer(self):
        self.tokenizer_summ = AutoTokenizer.from_pretrained("cointegrated/rut5-base-multitask")
        self.model_summ = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-base-multitask")


    
    