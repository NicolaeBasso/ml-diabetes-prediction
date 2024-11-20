import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
