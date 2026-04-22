from functools import lru_cache
import os
import pdfplumber

class TextPreprocessor:
    @lru_cache(maxsize=8)
    def extract_text_from_pdf(self, path: str) -> str:
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    def extract(self, path: str) -> str:
        stat = os.stat(path)
        return self.extract_text_from_pdf(path, stat.st_mtime, stat.st_size)

    @lru_cache(maxsize=32)
    def split_into_sections(self, text: str, section_size: int = 30) -> tuple[str, ...]:
        lines = text.split("\n")
        return tuple(
            " ".join(lines[i:i + section_size])
            for i in range(0, len(lines), section_size)
            if len(" ".join(lines[i:i + section_size]).strip()) > 0
        )

    @lru_cache(maxsize=64)
    def preprocess_text(self, text: str) -> str:
        lines = (
            line.strip().lower()
            for line in text.split("\n")
            if len(line.strip()) > 30
        )
        return " ".join(lines)



