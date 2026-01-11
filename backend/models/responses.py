from typing import List, Union, Optional
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    style_result: str
    coherence_result: Union[float, dict]
    coherence_interpretation: str
    structure_result: Union[float, dict]
    structure_interpret: str
    read_result: Union[float, dict]
    keywords: List[str]
    recommendation: Optional[str] = None

    @property
    def summary(self) -> str:
        """Краткое текстовое резюме анализа для LLM"""
        def to_float(value, key=None):
            if isinstance(value, dict) and key in value:
                return float(value[key])
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                return 0.0

        coherence = to_float(self.coherence_result)
        structure = to_float(self.structure_result)
        read = to_float(self.read_result, key="score")

        return (
            f"Style: {self.style_result}, "
            f"Coherence: {coherence:.1f}% ({self.coherence_interpretation}), "
            f"Structure: {structure:.1f}% ({self.structure_interpret}), "
            f"Readability: {read:.1f}%, "
            f"Keywords: {', '.join(self.keywords)}"
        )