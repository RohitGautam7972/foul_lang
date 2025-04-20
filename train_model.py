import pandas as pd  # <-- Add this
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load TF-IDF features and labels
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

df = pd.read_csv('preprocessed_twitter_data.csv')
X = vectorizer.transform(df['text'])  # Transform text to TF-IDF
y = df['label']

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
