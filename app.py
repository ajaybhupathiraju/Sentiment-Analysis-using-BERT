from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
from transformers import TFBertForSequenceClassification, BertTokenizerFast
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
            model = TFBertForSequenceClassification.from_pretrained("./custom_bert", num_labels=2)
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

            inputs = tokenizer([text.review_text], padding=True, return_tensors="tf")
            logits = model(**inputs).logits
            result = np.argmax(logits)
            return "review is -> {}".format(class_names[result])
    except Exception as e:
        raise e

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
