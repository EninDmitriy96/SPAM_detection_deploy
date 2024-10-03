from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

with open('/app/models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('/app/models/spam_model.pkl', 'rb') as f:
    model = pickle.load(f)


class Message(BaseModel):
    text: str


@app.post("/predict")
def predict_spam(message: Message):
    # Vectorize the input message
    input_vector = vectorizer.transform([message.text])

    # Predict if it's spam or not
    prediction = model.predict(input_vector)[0]

    return {"prediction": prediction}
