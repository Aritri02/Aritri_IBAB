##Bayesian Model implementation##
##spam_sms.csv is used for classification of spam and not spam ##
##Sklearn implementation##

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, encoding='latin-1')  # or can use encoding='ISO-8859-1' if needed
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def train_naive_bayes(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label_num'], test_size=0.3, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)

    y_pred = nb.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])

    return acc, report

def main():
    filepath = '/home/ibab/Downloads/spam_sms.csv'
    df = load_and_prepare_data(filepath)
    accuracy, report = train_naive_bayes(df)
    print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n")
    print(report)

if __name__ == "__main__":
    main()
