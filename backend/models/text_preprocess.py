import pdfplumber

class TextPreprocessor:
    def extract_text_from_pdf(self, path: str) -> str:
        """Извлекает текст из PDF по указанному пути."""
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    def split_into_sections(self, text: str, section_size: int = 30):
        """Делит текст на секции по количеству строк."""
        lines = text.split('\n')
        return [' '.join(lines[i:i+section_size]) for i in range(0, len(lines), section_size)]

