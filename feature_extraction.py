import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle  # To save the vectorizer for later use

# Load preprocessed data
df = pd.read_csv('preprocessed_twitter_data.csv')

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 most important words
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Save vectorizer for future use
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("TF-IDF feature extraction complete!")
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
