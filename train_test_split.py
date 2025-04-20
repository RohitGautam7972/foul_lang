import pandas as pd
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (if not already downloaded)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocess text by lowercasing, removing special characters, and filtering stopwords."""
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters & numbers
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

def prepare_data(data_path):
    """Load, preprocess, and split data into train/test sets."""
    df = pd.read_csv(data_path)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("DataFrame must contain 'text' and 'label' columns")
    
    df['processed_text'] = df['text'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

# Run the function
X_train, X_test, y_train, y_test = prepare_data('preprocessed_twitter_data.csv')

# Display sample processed data
print("Training data samples:")
print(X_train.head())
