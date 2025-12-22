import pandas as pd
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

from base import BaseAnalyzer
from sentiment import DictBasedSentiment
import config


class LightAnalyzer(BaseAnalyzer):
    """
    Анализатор тематических блоков (Light режим).
    """
    
    def __init__(self):
        super().__init__()
        print("[Light] ⚡ Инициализация Light-анализатора...")
        self.sentiment_analyzer = DictBasedSentiment()
        self.kw_extractor = yake.KeywordExtractor(lan="ru", n=1, top=5)
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'это']
        )

    def analyze(self, text: str, **kwargs) -> pd.DataFrame:
        min_len = kwargs.get('min_length', config.MIN_SENTENCE_LENGTH)
        n_clusters_cfg = kwargs.get('n_clusters', config.LIGHT_N_CLUSTERS)
        
        sentences = self.segmenter.segment(text, min_length=min_len)
        if not sentences:
            return pd.DataFrame()
        
        try:
            X = self.vectorizer.fit_transform(sentences)
        except ValueError:
            return pd.DataFrame()
        
        n_clusters = min(n_clusters_cfg, len(sentences))
        kmeans = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
        topics = kmeans.fit_predict(X)
        
        results = []
        for topic_id in sorted(set(topics)):
            topic_indices = [i for i, t in enumerate(topics) if t == topic_id]
            topic_sentences = [sentences[i] for i in topic_indices]
            combined_text = " ".join(topic_sentences)
            
            sentiment_result = self.sentiment_analyzer.analyze(combined_text)
            
            first_sent = topic_sentences[0][:50] + "..." if len(topic_sentences[0]) > 50 else topic_sentences[0]
            last_sent = topic_sentences[-1][:50] + "..." if len(topic_sentences[-1]) > 50 else topic_sentences[-1]
            summary = f"{first_sent} ... {last_sent}" if len(topic_sentences) > 1 else first_sent
            
            try:
                keywords_list = [kw[0] for kw in self.kw_extractor.extract_keywords(combined_text)]
                keywords = ", ".join(keywords_list[:5])
            except:
                keywords = ""
            
            topic_vectors = X[topic_indices].toarray() if hasattr(X, 'toarray') else X[topic_indices]
            center = np.mean(topic_vectors, axis=0)
            
            similarities = []
            for vec in topic_vectors:
                dot = np.dot(center, vec)
                norm = np.linalg.norm(center) * np.linalg.norm(vec)
                similarities.append(dot / norm if norm > 0 else 0)
            
            key_sent = topic_sentences[np.argmax(similarities)]
            key_sentence = key_sent[:80] + "..." if len(key_sent) > 80 else key_sent
            
            percentage = round(len(topic_sentences) / len(sentences) * 100, 1)
            
            results.append({
                "topic_id": topic_id,
                "theme": f"Тема {topic_id + 1}",
                "sentiment": sentiment_result["sentiment"],
                "top_emotion": "",      
                "emotions_json": {},    
                "summary": summary,
                "keywords": keywords,
                "key_sentence": key_sentence,
                "percentage": f"{percentage}%",
                "n_sentences": len(topic_sentences),
            })
        
        return pd.DataFrame(results)
