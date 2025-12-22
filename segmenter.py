import re
from typing import List

class TextSegmenter:
    """
    Класс для разбиения текста на предложения.
    Умеет обходить сокращения, чтобы не разрывать их как конец предложения.
    """
    
    ABBREVIATIONS = {
        'т.е', 'т.к', 'и.т.д', 'и.т.п', 'др', 'см', 'гл', 'стр',
        'ед', 'ст', 'п', 'пп', 'подп', 'св', 'пр'
    }

    @staticmethod
    def segment(text: str, min_length: int = 10) -> List[str]:
        """
        Разбивает текст на список предложений.
        
        Args:
            text: Исходный текст
            min_length: Минимальная длина предложения (мусорные короткие фразы откидываем)
        """
        protected_text = text
        
        for abbr in TextSegmenter.ABBREVIATIONS:
            protected_text = protected_text.replace(abbr, abbr.replace('.', '§'))
        
        sentences = re.split(r'(?<=[.!?])\s+', protected_text)
        
        clean_sentences = []
        for s in sentences:
            s_restored = s.replace('§', '.').strip()
            if len(s_restored) >= min_length:
                clean_sentences.append(s_restored)

        return clean_sentences if clean_sentences else [text]
