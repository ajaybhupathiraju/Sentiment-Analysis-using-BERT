from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import tensorflow as tf
import numpy as np
from transformers import AutoImageProcessor, TFViTModel,TFBertForSequenceClassification
import keras
from keras.layers import Dense, Flatten
import warnings

warnings.filterwarnings("ignore")

# define fastapi instance
app = FastAPI()

# predicted classes
class_names = ["negative", "positive"]


class ReviewText(BaseModel):
    review_text: str


@app.post("/review/")
async def predict_review(text: ReviewText):
    print(type(text.review_text))
    try:
        if text is None or text == "" or text.review_text == "" or text.review_text is None:
            raise HTTPException(status_code=422, detail="review text should not be empty or none")
        else:
            input = text.review_text
            model = TFBertForSequenceClassification.from_pretrained("./custom_bert", num_labels=2)
            print("model :{}".format(model))
            return ""
    except Exception as e:
        raise e


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
