# -*- coding: utf-8 -*-
import re
from typing import Dict, List
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import config

class DictBasedSentiment:
    """Быстрый анализ тональности (Light)."""
    def __init__(self):
        self.positive_words = {
            "рост", "успех", "прибыль", "развитие", "хорошо", "плюс", "выгода", "стабильно",
            "успешно", "отличный", "прекрасно", "позитив", "улучшение", "достижение",
            "впечатляющий", "замечательный", "превосходный", "восхитительно", "блестяще",
            "потрясающе", "гениально", "шедевр", "восторг", "радость", "счастье", "любовь",
        }
        self.negative_words = {
            "убыток", "кризис", "падение", "долг", "риск", "санкции", "проблема", "минус",
            "провал", "плохо", "ужасно", "негатив", "ухудшение", "потеря", "опасность",
            "критический", "бедственный", "разочарование", "разочаровал", "кошмар",
            "катастрофа", "отвратительно", "страх", "боль", "страдание",
        }

    def analyze(self, text: str) -> Dict[str, float]:
        words = set(re.findall(r"\w+", text.lower()))
        if not words:
            return {"sentiment": "neutral", "score": 0.0}
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        raw = pos_count - neg_count
        score = max(-1.0, min(1.0, raw / max(1, len(words)) * 3))
        if score > 0.10: label = "positive"
        elif score < -0.10: label = "negative"
        else: label = "neutral"
        return {"sentiment": label, "score": round(float(score), 3)}


class BertSentiment:
    """Базовый сентимент (Heavy) +/-."""
    def __init__(self, model_name: str = config.SENTIMENT_MODEL_RU):
        device = 0 if torch.cuda.is_available() else -1
        print(f"[Sentiment] Загрузка модели {model_name}...")
        self.pipe = pipeline("sentiment-analysis", model=model_name, device=device)
        self.label_map = {
            "LABEL_0": "neutral", "LABEL_1": "positive", "LABEL_2": "negative",
            "NEUTRAL": "neutral", "POSITIVE": "positive", "NEGATIVE": "negative"
        }

    def analyze(self, text: str) -> Dict[str, float]:
        try:
            result = self.pipe(text[:512])[0]
            label = self.label_map.get(result["label"], "neutral")
            score = float(result["score"])
            norm_score = score if label == "positive" else (-score if label == "negative" else 0.0)
            return {"sentiment": label, "score": round(norm_score, 3)}
        except:
            return {"sentiment": "neutral", "score": 0.0}


class EmotionClassifier:
    """
    Детальный анализ эмоций (Heavy).
    Использует rubert-tiny2 (Aniemore/rubert-tiny2-russian-emotion-detection).
    """
    def __init__(self, model_name: str = config.EMOTION_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Emotion] Загрузка Tiny-модели эмоций {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> Dict[str, float]:
        """Возвращает вероятности для каждой эмоции."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Применяем softmax для классификации (сумма = 1)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            result = {}
            for idx, prob in enumerate(probs):
                label = self.id2label.get(idx, str(idx))
                ru_map = {
                    "anger": "гнев", "fear": "страх", "happiness": "радость", "joy": "радость",
                    "sadness": "грусть", "surprise": "удивление", "neutral": "нейтрально",
                    "enthusiasm": "энтузиазм", "disgust": "отвращение", 
                    "guilt": "вина", "shame": "стыд"
                }
                label_ru = ru_map.get(label, label)
                result[label_ru] = round(float(prob), 3)
            
            return result
        except Exception as e:
            print(f"Emotion error: {e}")
            return {}
