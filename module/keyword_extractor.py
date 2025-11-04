from dataclasses import dataclass
import yake

@dataclass
class ExtractorConfig:
    lan: str
    dedupLim: float
    dedupFunc: str
    windowsSize: int

class KeywordExtractor:
    def __init__(self, config: ExtractorConfig):
        self.config = config

    def extract(self, raw_text, top, n):
        extractor = yake.KeywordExtractor(
            top=top,
            n=n,
            lan=self.config.lan,
            dedupFunc=self.config.dedupFunc,
            dedupLim=self.config.dedupLim,
            windowsSize=self.config.windowsSize
        )
        return extractor.extract_keywords(raw_text)
