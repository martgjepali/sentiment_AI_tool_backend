import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Make sure to download these once before using them
nltk.download('punkt')
nltk.download('stopwords')

def load_dataset(filepath):
    print("Loading dataset from:", filepath)
    # Assuming your data is tab-separated and the last column is the label
    df = pd.read_csv(filepath, sep='\t', header=None)
    df.columns = ['text', 'sentiment']
    print("Loaded dataset shape:", df.shape)
    print("First few rows of the dataset:")
    print(df.head())
    return df

def preprocess_text(text: str):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Convert to lower case
    tokens = [w.lower() for w in tokens]
    # Remove punctuation and non-alphabetic characters
    words = [word for word in tokens if word.isalpha()]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return " ".join(words)

def train_sentiment_model(df):
    # Preprocess the text
    df['preprocessed'] = df['text'].apply(preprocess_text)

    # Vectorize the preprocessed text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['preprocessed'])
    y = df['sentiment']

    # Split the dataset into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, vectorizer

def save_model(model, vectorizer, model_path='../../models/sentiment_model.pkl', vectorizer_path='../../models/vectorizer.pkl'):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    # Define the path to your dataset
    dataset_path = 'amazon_cells_labelled.txt'
    # Load the dataset
    df = load_dataset(dataset_path)
    # Train the sentiment analysis model
    model, vectorizer = train_sentiment_model(df)
    # Save the model and vectorizer to disk
    save_model(model, vectorizer)
