from abc import ABC, abstractmethod
import pandas as pd
from segmenter import TextSegmenter

class BaseAnalyzer(ABC):
    """
    Базовый родительский класс.
    Определяет структуру, которой должны следовать Light и Heavy анализаторы.
    """
    def __init__(self):
        self.segmenter = TextSegmenter()

    @abstractmethod
    def analyze(self, text: str, **kwargs) -> pd.DataFrame:
        """
        Метод анализа.
        kwargs позволяет передавать динамические параметры (n_clusters, threshold и т.д.)
        """
        pass