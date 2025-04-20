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

    # Convert probability to percentage
    percentage = round(probability * 100, 2)

    # Categorizing the severity with colored output
    if probability < 0.4:
        severity = colored("🟢 Low Risk", "green")
        final_decision = colored("✅ Clean Language", "green", attrs=["bold"])
    elif 0.4 <= probability < 0.7:
        severity = colored("🟡 Moderate Risk", "yellow")
        final_decision = colored("✅ Clean Language (Moderate Risk Allowed)", "green", attrs=["bold"])
    else:
        severity = colored("🔴 High Risk", "red")
        final_decision = colored("❌ Foul Language Detected (Rejected)", "red", attrs=["bold"])

    # Extract top words contributing to the foul language prediction
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(text_vectorized.toarray()).flatten()[::-1]
    top_words = feature_array[tfidf_sorting][:5]  # Get top 5 words

    return {
        "Complaint Text": complaint,
        "Processed Text": processed_text,
        "Foul Language Probability": f"{percentage}%",  # Show percentage
        "Severity Level": severity,
        "Top Contributing Words": list(top_words),
        "Final Decision": final_decision
    }

# Predefined foul complaint for testing
foul_complaint = """What the hell is going on with the recent math exam? It was a complete disaster! Half the questions were messed up, and whoever wrote this crap should be fired. Question number 5 made no damn sense, and question number 12 had a stupid typo that screwed everything up.
How the heck did this even get approved? This is such a joke, and it’s unfair to all the students who worked their butts off to prepare. Fix this mess ASAP, or I’m going to raise hell about it."""

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
print(colored("🚨 Final Decision:", "red", attrs=["bold"]), result["Final Decision"])
print("=" * 50 + "\n")
