import pandas as pd
from preprocessing import preprocess_text, vectorize_data
from modeling import train_evaluate_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/spam.csv', encoding='latin-1', dtype=str)


df = df[['Category', 'Message']]
df.columns = ['label', 'text']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

df['cleaned_text'] = df['text'].apply(preprocess_text)

X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_tfidf, X_test_tfidf, tfidf = vectorize_data(X_train, X_test)

model, X_test_tfidf, y_test, y_pred = train_evaluate_model(X_train_tfidf, y_train)
