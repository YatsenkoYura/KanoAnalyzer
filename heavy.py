from typing import List
import numpy as np
import pandas as pd
import torch
import yake
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering

from base import BaseAnalyzer
from sentiment import BertSentiment, EmotionClassifier
import config

class HeavyAnalyzer(BaseAnalyzer):
    """
    - LaBSE эмбеддинги (правильный pooling)
    - Agglomerative clustering (динамический порог)
    - T5 Summary + Emotions
    """

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("[Heavy] 1/4 Загрузка LaBSE...")
        self.labse_tokenizer = AutoTokenizer.from_pretrained(config.LABSE_MODEL)
        self.labse_model = AutoModel.from_pretrained(config.LABSE_MODEL).to(self.device).eval()

        print("[Heavy] 2/4 Загрузка сентимента...")
        self.sentiment_model = BertSentiment()
        
        print("[Heavy] 3/4 Загрузка анализатора эмоций...")
        self.emotion_classifier = EmotionClassifier()
        
        print("[Heavy] 4/4 Загрузка суммаризатора (T5)...")
        self.sum_tokenizer = AutoTokenizer.from_pretrained(config.SUMMARY_MODEL)
        self.sum_model = T5ForConditionalGeneration.from_pretrained(config.SUMMARY_MODEL).to(self.device)

        self.kw_extractor = yake.KeywordExtractor(lan="ru", n=1, top=5)

    @torch.no_grad()
    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        embeddings = []
        for i in range(0, len(sentences), config.BATCH_SIZE_ENCODING):
            batch = sentences[i : i + config.BATCH_SIZE_ENCODING]
            tokens = self.labse_tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            output = self.labse_model(**tokens)
            last_hidden = output.last_hidden_state
            
            attention_mask = tokens.attention_mask.unsqueeze(-1)
            sum_embeddings = (last_hidden * attention_mask).sum(1)
            sum_mask = attention_mask.sum(1).clamp(min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
            
            embeddings.append(mean_embeddings.cpu().numpy())
        return np.vstack(embeddings).astype(np.float32)

    def generate_summary(self, text: str) -> str:
        try:
            input_ids = self.sum_tokenizer(text, return_tensors="pt", max_length=600, truncation=True).input_ids.to(self.device)
            output_ids = self.sum_model.generate(
                input_ids, max_length=150, min_length=30, 
                no_repeat_ngram_size=3, num_beams=2, early_stopping=True
            )
            return self.sum_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Summary error: {e}")
            return text[:200] + "..."

    def analyze(self, text: str, **kwargs) -> pd.DataFrame:
        threshold = kwargs.get('threshold', config.HEAVY_DISTANCE_THRESHOLD)
        min_len = kwargs.get('min_length', config.MIN_SENTENCE_LENGTH)

        sentences = self.segmenter.segment(text, min_length=min_len)
        if not sentences: return pd.DataFrame()

        print(f"[Heavy] Кодирование {len(sentences)} предложений...")
        embeddings = self.encode_sentences(sentences)

        print(f"[Heavy] Кластеризация (порог={threshold})...")
        distances = cosine_distances(embeddings)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="precomputed",
            linkage="average",
        )
        topics = clustering.fit_predict(distances)

        results = []
        total = len(sentences)

        for topic_id in sorted(set(topics)):
            idxs = [i for i, t in enumerate(topics) if t == topic_id]
            topic_sentences = [sentences[i] for i in idxs]
            topic_embeddings = embeddings[idxs]
            combined_text = " ".join(topic_sentences)

            try:
                kws = [kw[0] for kw in self.kw_extractor.extract_keywords(combined_text)]
                keywords = ", ".join(kws[:5])
            except: keywords = ""

            center = np.mean(topic_embeddings, axis=0)
            d = cosine_distances(topic_embeddings, center.reshape(1, -1)).reshape(-1)
            key_sentence = topic_sentences[int(np.argmin(d))]

            sent = self.sentiment_model.analyze(combined_text)
            emotions = self.emotion_classifier.predict(combined_text)
            top_emotion = max(emotions, key=emotions.get) if emotions else "нейтрально"

            if len(topic_sentences) >= 3:
                summary = self.generate_summary(combined_text)
            else:
                summary = combined_text

            percentage = round(len(topic_sentences) / total * 100.0, 1)

            results.append({
                "topic_id": int(topic_id),
                "theme": f"Тема {int(topic_id) + 1}",
                "sentiment": sent["sentiment"],
                "top_emotion": top_emotion,
                "emotions_json": emotions,
                "summary": summary,
                "keywords": keywords,
                "key_sentence": key_sentence,
                "percentage": f"{percentage}%",
                "n_sentences": int(len(topic_sentences)),
            })

        return pd.DataFrame(results).sort_values("topic_id").reset_index(drop=True)
