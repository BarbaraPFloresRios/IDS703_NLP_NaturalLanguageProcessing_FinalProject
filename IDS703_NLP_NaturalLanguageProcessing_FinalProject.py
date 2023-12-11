"""Compare token/document vectors for classification."""
import random
from typing import Mapping, Optional, Sequence
import nltk
import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression


FloatArray = NDArray[np.float64]


def onehot(
    vocabulary_map: Mapping[Optional[str], int], token: Optional[str]
) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary_map),))
    idx = vocabulary_map.get(token, len(vocabulary_map) - 1)
    embedding[idx] = 1
    return embedding


def sum_token_embeddings(
    token_embeddings: Sequence[FloatArray],
) -> FloatArray:
    """Sum the token embeddings."""
    total: FloatArray = np.array(token_embeddings).sum(axis=0)
    return total


def split_train_test(
    X: FloatArray, y: FloatArray, test_percent: float = 10
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Split data into training and testing sets."""
    N = len(y)
    data_idx = list(range(N))
    random.shuffle(data_idx)
    break_idx = round(test_percent / 100 * N)
    training_idx = data_idx[break_idx:]
    testing_idx = data_idx[:break_idx]
    X_train = X[training_idx, :]
    y_train = y[training_idx]
    X_test = X[testing_idx, :]
    y_test = y[testing_idx]
    return X_train, y_train, X_test, y_test


def generate_data_token_counts(
    h0_documents: list[list[str]], h1_documents: list[list[str]]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with raw token counts."""
    X: FloatArray = np.array(
        [
            sum_token_embeddings([onehot(vocabulary_map, token) for token in sentence])
            for sentence in h0_documents
        ]
        + [
            sum_token_embeddings([onehot(vocabulary_map, token) for token in sentence])
            for sentence in h1_documents
        ]
    )
    y: FloatArray = np.array(
        [0 for sentence in h0_documents] + [1 for sentence in h1_documents]
    )
    return split_train_test(X, y)


def build_vocabulary(h0_documents, h1_documents):
    vocabulary = sorted(
        set(token for sentence in h0_documents + h1_documents for token in sentence)
    ) + [None]
    vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}
    return vocabulary_map


## 2A: Generative Probabilistic Model: Naive_bayes


def train_naive_bayes(X_train, y_train):
    # ph0: Probability of class 0
    # ph1: Probability of class 1
    # p0: Matrix of probabilities for each token in the vocabulary given class 0
    # p1: Matrix of probabilities for each token in the vocabulary given class 1

    h0_documents = X_train[y_train == 0]
    h1_documents = X_train[y_train == 1]

    ph0 = len(h0_documents) / len(y_train)
    ph1 = len(h1_documents) / len(y_train)

    counts0 = np.sum(h0_documents, axis=0) + 1
    counts1 = np.sum(h1_documents, axis=0) + 1

    p0 = counts0 / counts0.sum()
    p1 = counts1 / counts1.sum()

    return ph0, ph1, p0, p1


def test_naive_bayes(X_test, y_test, ph0, ph1, p0, p1):
    num_correct = 0
    for i in range(len(y_test)):
        sentence = X_test[i]
        label = y_test[i]
        h0_logp_unnormalized = sentence.T @ np.log(p0) + np.log(ph0)
        h1_logp_unnormalized = sentence.T @ np.log(p1) + np.log(ph1)

        # normalize
        logp_data = np.logaddexp(h0_logp_unnormalized, h1_logp_unnormalized)
        h0_logp = h0_logp_unnormalized - logp_data
        h1_logp = h1_logp_unnormalized - logp_data

        # make guess
        pc0 = np.exp(h0_logp)
        pc1 = np.exp(h1_logp)
        guess = 1 if pc1 > pc0 else 0

        if guess == label:
            num_correct += 1

    return num_correct / len(y_test)


## 2B: Discriminative Neural Network: Logistic Regression


def train_logistic_regression_with_tfidf(X_train, y_train):
    # Train TF-IDF
    tfidf = TfidfTransformer(norm=None).fit(X_train)
    X_train_tfidf = tfidf.transform(X_train)

    # Train Logistic Regression
    clf = LogisticRegression(random_state=0, max_iter=3000, solver="sag").fit(
        X_train_tfidf, y_train
    )

    return clf, tfidf, X_train_tfidf


def test_logistic_regression_with_tfidf(X_test, y_test, clf, tfidf):
    # Transform X_test using TF-IDF
    X_test_tfidf = tfidf.transform(X_test)

    # Predict using the trained logistic regression model
    predictions = clf.predict(X_test_tfidf)

    # Calculate accuracy
    accuracy = sum(predictions == y_test) / len(y_test)

    return accuracy


# Load Data
h0_documents = nltk.corpus.gutenberg.sents("shakespeare-hamlet.txt")
h1_documents = nltk.corpus.gutenberg.sents("bible-kjv.txt")[:3106]
h0_documents = h0_documents[: max(len(h0_documents), len(h1_documents))]
h1_documents = h1_documents[: max(len(h0_documents), len(h1_documents))]

# Print information about the documents
print("Document Information:\n")
print("h0 = Hamlet by Shakespeare")
print("Number of sentences in h0:", len(h0_documents))
print(
    "Number of unique words in h0:",
    len(set(token for sentence in h0_documents for token in sentence)),
)
print()

print("h1 = Bible")
print("Number of sentences in h1:", len(h1_documents))
print(
    "Number of unique words in h1:",
    len(set(token for sentence in h1_documents for token in sentence)),
)
print()

vocabulary_map = build_vocabulary(h0_documents, h1_documents)
X_train, y_train, X_test, y_test = generate_data_token_counts(
    h0_documents, h1_documents
)

# Train Naive Bayes
ph0, ph1, p0, p1 = train_naive_bayes(X_train, y_train)

# Test Naive Bayes
print("Results:\n")
print("Naive Bayes (train):", test_naive_bayes(X_train, y_train, ph0, ph1, p0, p1))
print("Naive Bayes (test):", test_naive_bayes(X_test, y_test, ph0, ph1, p0, p1))
print()

# Train Logistic Regression with TF-IDF
clf_logistic, tfidf_logistic, X_train_tfidf = train_logistic_regression_with_tfidf(
    X_train, y_train
)

# Test Logistic Regression with TF-IDF
accuracy_logistic_train = test_logistic_regression_with_tfidf(
    X_train, y_train, clf_logistic, tfidf_logistic
)
accuracy_logistic_test = test_logistic_regression_with_tfidf(
    X_test, y_test, clf_logistic, tfidf_logistic
)

print("Logistic Regression with TF-IDF (train):", accuracy_logistic_train)
print("Logistic Regression with TF-IDF (test):", accuracy_logistic_test)
