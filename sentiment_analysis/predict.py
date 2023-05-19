from typing import List, Dict
import numpy as np

def custom_predict(y_prob, threshold, index):
    """Custom predict function that defaults
    to an index if conditions are not met."""
    y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
    return np.array(y_pred)

def predict(texts: List, artifacts: Dict) -> List:
    """Predict sentiments for given texts.

    Args:
        texts (List): raw input texts to classify.
        artifacts (Dict): artifacts from a run.

    Returns:
        List: predictions for input texts.
    """
    x = artifacts["vectorizer"].transform(texts)
    y_pred = custom_predict(
        y_prob=artifacts["model"].predict_proba(x),
        threshold=artifacts["args"].threshold,
        index=artifacts["label_encoder"].class_to_index["neutral"],
    )
    sentiments = artifacts["label_encoder"].decode(y_pred)
    predictions = [
        {
            "input_text": texts[i],
            "predicted_sentiment": sentiments[i],
        }
        for i in range(len(sentiments))
    ]
    return predictions