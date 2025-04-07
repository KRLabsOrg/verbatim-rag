import pandas as pd
import string
import unicodedata
import re

def preprocess_pubmedqa(train_path: str, test_path: str, output_train_path: str, output_test_path: str):
    def clean_whitespace(text):
        return re.sub(r'\s+', ' ', text).strip()

    def normalize_unicode(text):
        return unicodedata.normalize("NFKC", text)

    def remove_punctuation(text):
        return text.translate(str.maketrans("", "", string.punctuation))

    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    for col in ["question", "sentence"]:
        # Ensure string type
        train_data[col] = train_data[col].astype(str).str.lower().apply(clean_whitespace).apply(normalize_unicode).apply(remove_punctuation)
        test_data[col] = test_data[col].astype(str).str.lower().apply(clean_whitespace).apply(normalize_unicode).apply(remove_punctuation)

    # Save processed files
    train_data.to_csv(output_train_path, index=False)
    test_data.to_csv(output_test_path, index=False)
