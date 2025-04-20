import pickle
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from termcolor import colored  # For colored terminal output

# Ensure NLTK stopwords are available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained model and vectorizer
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters & numbers
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Function to check complaint
def check_complaint(complaint):
    processed_text = preprocess_text(complaint)
    text_vectorized = vectorizer.transform([processed_text])
    probability = model.predict_proba(text_vectorized)[0][1]  # Probability of foul language
    prediction = model.predict(text_vectorized)[0]

    # Categorizing the severity with colored output
    if probability < 0.4:
        severity = colored("🟢 Low Risk", "green")
        final_decision = "✅ No Foul Language Detected"
    elif 0.4 <= probability < 0.7:
        severity = colored("🟡 Moderate Risk", "yellow")
        final_decision = "⚠️ Possible Foul Language, Needs Review"
    else:
        severity = colored("🔴 High Risk", "red")
        final_decision = "❌ Foul Language Detected"

    # Extract top words contributing to the foul language prediction
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(text_vectorized.toarray()).flatten()[::-1]
    top_words = feature_array[tfidf_sorting][:5]  # Get top 5 words

    return {
        "Complaint Text": complaint,
        "Processed Text": processed_text,
        "Foul Language Probability": round(probability, 4),
        "Severity Level": severity,
        "Top Contributing Words": list(top_words),
        "Final Decision": final_decision
    }

# Predefined foul complaint for testing
foul_complaint = """I can’t believe how utterly pathetic your service is! This is the worst experience I’ve ever had. 
You guys are completely useless, don’t know how to do your damn job, and clearly don’t give a crap about customers. 
I’ve been fucked waiting for my order for over two weeks, and all I get are lame excuses. 
What kind of incompetent idiots are running this place? Absolute garbage! 
I demand a refund right now, or I’ll make sure everyone knows how terrible this company is. 
Get your act together, you morons!"""

# Check predefined foul complaint
result = check_complaint(foul_complaint)

# Display detailed output with formatting
print("\n" + "=" * 50)
print(colored("📌 Complaint Analysis Report", "cyan", attrs=["bold"]))
print("=" * 50 + "\n")
print(colored("📝 Complaint Text:", "blue", attrs=["bold"]))
print(colored(result["Complaint Text"], "white") + "\n")
print(colored("📌 Processed Text:", "blue", attrs=["bold"]))
print(colored(result["Processed Text"], "white") + "\n")
print(colored("🔢 Foul Language Probability:", "blue", attrs=["bold"]), colored(result["Foul Language Probability"], "magenta"))
print(colored("⚠ Severity Level:", "blue", attrs=["bold"]), result["Severity Level"])
print(colored("🔥 Top Contributing Words:", "blue", attrs=["bold"]), colored(", ".join(result["Top Contributing Words"]), "yellow"))
print("=" * 50 + "\n")
print(colored("🚨 Final Decision:", "red", attrs=["bold"]), colored(result["Final Decision"], "cyan", attrs=["bold"]))
print("\n" + "=" * 50)
