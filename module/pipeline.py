from module.keyword_extractor import KeywordExtractor, ExtractorConfig
from module.text_processor import TextProcessor

class Pipeline:
    def __init__(self, extractor_config : ExtractorConfig):
        self.text_processor = TextProcessor()
        self.extractor_config = extractor_config
        self.keyword_extractor = KeywordExtractor(self.extractor_config)

    def run_text_processor(self, raw_text : str) -> list[str]:
        return self.text_processor.apply_text_processing(raw_text)

    def run_extract_keyword(self, raw_text : str, top : int, n: int) -> list[str]:
        return self.keyword_extractor.extract(raw_text, top, n)
    
        

