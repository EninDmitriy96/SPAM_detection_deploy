import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import mlflow.sklearn
import pickle
import mlflow


def train_save_model():
    with mlflow.start_run():
        X_train = pd.read_csv('data/processed/X_train.csv')
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_train = pd.read_csv('data/processed/y_train.csv')
        y_test = pd.read_csv('data/processed/y_test.csv')

        vectorizer = CountVectorizer()

        x_train_vec = vectorizer.fit_transform(X_train['Message'])
        x_test_vec = vectorizer.transform(X_test['Message'])

        model = MultinomialNB()
        model.fit(x_train_vec, y_train['Class'])

        y_pred = model.predict(x_test_vec)
        acc = accuracy_score(y_test['Class'], y_pred)

        with open('models/spam_classifier_model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        with open('models/vectorizer.pkl', 'wb') as vec_file:
            pickle.dump(vectorizer, vec_file)

        print(f'Accuracy: {acc:.2f}')
        mlflow.log_artifact('models/spam_classifier_model.pkl')
        mlflow.log_artifact('models/vectorizer.pkl')
        mlflow.sklearn.log_model(model, 'MultinomialNB_spam_classifier_model')


if __name__ == '__main__':
    train_save_model()
