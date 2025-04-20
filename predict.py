import pickle
import pandas as pd

# Load the trained model and vectorizer
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to predict new tweets
def predict_tweet(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    label = "Foul Language" if prediction == 1 else "Non-Foul Language"
    return label

# Test a new tweet
new_tweet = "I can’t believe how utterly pathetic your service is! 
This is the worst experience I’ve ever had. 
You guys are completely useless, don’t know how to do your damn job, and clearly don’t give a crap about customers. 
I’ve been waiting for my order for over two weeks, and all I get are lame excuses. 
What kind of incompetent idiots are running this place? Absolute garbage! 
I demand a refund right now, or I’ll make sure everyone knows how terrible this company is. 
Get your act together, you morons!
"
print(f"Tweet: {new_tweet}")
print(f"Prediction: {predict_tweet(new_tweet)}")
