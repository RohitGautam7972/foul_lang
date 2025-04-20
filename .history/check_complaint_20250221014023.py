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
foul_complaint = """Dear Principal [Last Name],

I am writing to bring to your attention a concerning issue that has been affecting my child, [Child’s Name], who is in [Grade/Class]. Over the past few weeks, I have noticed a consistent problem with [specific issue, e.g., the lack of supervision during lunch breaks, inadequate communication about assignments, or recurring bullying incidents].

For example, [provide a specific incident or detail, e.g., "Last Tuesday, my child came home upset after being repeatedly teased by a group of students during recess. Despite reporting this to a teacher, the behavior has continued."]. This has not only impacted my child’s emotional well-being but has also made it difficult for them to focus on their studies.

I understand that managing a school is a complex task, and I truly appreciate the efforts you and your staff make to create a safe and nurturing environment. However, I believe this issue requires immediate attention to ensure that all students feel respected and supported.

I kindly request that you look into this matter and take appropriate steps to address it. Whether it’s increased supervision, a meeting with the involved students, or a school-wide initiative to promote kindness and respect, I trust your judgment in finding a solution.

Thank you for your time and understanding. I am happy to discuss this further if needed and am hopeful that we can work together to resolve this issue.

Sincerely,"""

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
