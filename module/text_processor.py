from nltk.corpus import stopwords
from pymystem3 import Mystem
import re
class TextProcessor:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.actual_text = self.raw_text

    def tokenize(self, text : str = None) -> list[str]:
        if text is None:
            target_text = self.actual_text
        else:
            target_text = text

        try:
            self.actual_text = target_text.split()
            return self.actual_text
        except Exception as e:
            print("tokenize error: " + e)
            
    def normalize(self, text : list[str] = None) -> list[str]:
        if text is None:
            target_text = self.actual_text
        else:
            target_text = text

        try:
            normaled_text = []
            for i in target_text:
                normaled_text.append(re.sub(r'[^а-яёА-ЯЁa-zA-Z]', '', i.lower()))
            self.actual_text = normaled_text

            return self.actual_text
        except Exception as e:
            print("normalize error: " + e)
        else:
            self.actual_text = self.normaled_text
    
    def erase_stop_word(self, text : list[str] = None) -> list[str]:
        stop_words = set(stopwords.words('russian'))
        if text is None:
            target_text = self.actual_text
        else:
            target_text = text

        try:
            self.actual_text = [word for word in target_text if word not in stop_words]
            return self.actual_text

        except Exception as e:
            print("erase st.opword error: " + e)

    def lematize(self, text : list[str] = None) -> list[str]:
        if text is None:
            target_text = self.actual_text
        else:
            target_text = text
        try:
            m = Mystem()
            target_text = " ".join(target_text)
            target_text = "".join(m.lemmatize(target_text))
            self.actual_text = target_text.split()

            return self.actual_text
        except Exception  as e:
            print("error: "+ e)
    
    def get_result(self) -> list[str]:
        return self.actual_text

    def apply_text_processing(self, text : list[str] = None) -> list[str]:
        if text is None:
            target_text = self.raw_text
        else:
            target_text = text
            
        return self.lematize(
            self.erase_stop_word(
                self.normalize(
                    self.tokenize(target_text)
                )
            )
        )