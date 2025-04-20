import pandas as pd
import re

def clean_text(text, is_hindi=False):
    """Clean text while preserving offensive terms"""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs and HTML
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'&\w+;', '', text)
    
    if is_hindi:
        # Keep Hindi characters and common Roman script
        text = re.sub(r'[^\u0900-\u097F\w\s]', ' ', text)
    else:
        # For English text
        text = re.sub(r'[^\w\s\'.]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def prepare_multilingual_data(twitter_file, hindi_file):
    """Combine Twitter and Hindi datasets"""
    # Load datasets
    twitter_data = pd.read_csv(twitter_file)
    hindi_data = pd.read_csv(hindi_file)
    
    # Process Twitter data
    twitter_processed = pd.DataFrame({
        'text': twitter_data['text'].apply(lambda x: clean_text(x, is_hindi=False)),
        'label': twitter_data['label']
    })
    
    # Process Hindi data
    hindi_processed = pd.DataFrame({
        'text': hindi_data['commentText'].apply(lambda x: clean_text(x, is_hindi=True)),
        'label': hindi_data['labelval']
    })
    
    # Combine datasets
    combined_data = pd.concat([twitter_processed, hindi_processed], ignore_index=True)
    
    # Remove empty texts and shuffle
    combined_data = combined_data[combined_data['text'].str.len() > 0]
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined_data