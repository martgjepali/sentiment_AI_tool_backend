from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .sentiment_analysis.analyzer import preprocess_text
import pickle
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str

def get_model_path(model_name):
    # Get the absolute path to the directory where main.py is located
    current_directory = os.path.dirname(__file__)
    # Construct the path to the model file
    return os.path.join(current_directory, '.', 'models', model_name)

def predict_sentiment(text: str) -> str:
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Get the full paths to the model and vectorizer
    model_path = get_model_path('sentiment_model.pkl')
    vectorizer_path = get_model_path('vectorizer.pkl')
    
    # Load the trained model and vectorizer
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    # Vectorize the preprocessed text
    vectorized_text = vectorizer.transform([preprocessed_text])
    # Predict sentiment
    prediction = model.predict(vectorized_text)

    # Map the numerical prediction to a string
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    return sentiment

@app.post("/sentiment/", response_model=SentimentResponse)
def get_sentiment(request: SentimentRequest):
    sentiment = predict_sentiment(request.text)
    return SentimentResponse(sentiment=sentiment)
