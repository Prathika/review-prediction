# app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Get the input review
    review = data.get('review')
    
    # Vectorize the input review
    review_vectorized = vectorizer.transform([review])
    
    # Make a prediction
    sentiment = model.predict(review_vectorized)
    
    # Return the result
    return jsonify({'sentiment': int(sentiment[0]), 'label': 'positive' if sentiment[0] == 1 else 'negative'})

if __name__ == '__main__':
    app.run(debug=True)
