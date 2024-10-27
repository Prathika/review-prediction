# model.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Sample dataset
data = {
    'review': [
        "I loved this movie, it was fantastic!",
        "What a great film, I enjoyed every minute.",
        "This was the worst movie I have ever seen.",
        "I didn't like the plot at all.",
        "An excellent film with a brilliant cast!",
        "Not my cup of tea, it was boring."
    ],
    'sentiment': [1, 1, 0, 0, 1, 0]  # 1 for positive, 0 for negative
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df['review']
y = df['sentiment']

# Vectorize the text
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
