import pandas as pd

# Load the dataset
df = pd.read_csv('twitter_data.csv')  # Replace with your dataset filename

# Combine hate_speech (class 0) and offensive_language (class 1) into a single class (1 for foul language)
df['label'] = df.apply(lambda row: 1 if row['class'] in [0, 1] else 0, axis=1)

# Extract only the 'tweet' and 'label' columns
df = df[['tweet', 'label']]

# Rename 'tweet' to 'text' for consistency
df.rename(columns={'tweet': 'text'}, inplace=True)

# Save the preprocessed dataset (optional)
df.to_csv('preprocessed_twitter_data.csv', index=False)

# Display the first few rows
print(df.head())