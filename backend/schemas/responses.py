from pydantic import BaseModel
from typing import List, Dict, Any

class AnalysisResponse(BaseModel):
    style_result: str
    coherence_result: float
    coherence_interpretation: str
    structure_result: float
    structure_interpret: str
    read_result: Dict[str, Any]
    keywords: List[str]
