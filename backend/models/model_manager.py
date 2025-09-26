import warnings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, T5ForConditionalGeneration, DistilBertTokenizer, AutoModel

warnings.filterwarnings('ignore')

class ModelManager:
    def __init__(self):
        self._init_bert()
        self._init_summarizer()
        self._init_style_classifier()

    def _init_bert(self):
        self.model_bert = SentenceTransformer("cointegrated/rubert-tiny")

    def _init_summarizer(self):
        self.tokenizer_summ = AutoTokenizer.from_pretrained("cointegrated/rut5-base-multitask")
        self.model_summ = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-base-multitask")

    def _init_style_classifier(self):
        self.tokenizer_style = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model_style = AutoModel.from_pretrained(
            "textattack/distilbert-base-uncased-imdb", trust_remote_code=True, local_files_only=False)


    
    