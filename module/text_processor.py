from pymystem3 import Mystem
import re

class TextProcessor:
    def __init__(self):
        self.actual_text = None
        self.stop_words = []
        with open('src/stopwords-ru.txt', 'r') as f:
            for line in f:
                stop_words.append(re.sub("\n", "", line))

    def tokenize(self, text : str) -> list[str]:
        target_text = text

        try:
            self.actual_text = target_text.split()
            return self.actual_text
        except Exception as e:
            print("tokenize error: " + e)
            
    def normalize(self, text : list[str]) -> list[str]:
        target_text = text

        try:
            normaled_text = []
            for i in target_text:
                normaled_text.append(re.sub(r'[^а-яёА-ЯЁ]', '', i.lower()))
            self.actual_text = normaled_text

            return self.actual_text
        except Exception as e:
            print("normalize error: " + e)
        else:
            self.actual_text = self.normaled_text
    
    def erase_stop_word(self, text : list[str]) -> list[str]:
        target_text = text
        try:
            self.actual_text = [word for word in target_text if word not in self.stop_words]
            return self.actual_text

        except Exception as e:
            print("erase st.opword error: " + e)

    def lemmatize(self, text : list[str]) -> list[str]:
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

    def apply_text_processing(self, text : str) -> list[str]:
        return self.lemmatize(
            self.erase_stop_word(
                self.normalize(
                    self.tokenize(text)
                )
            )
        )