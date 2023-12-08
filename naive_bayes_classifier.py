"""Compare token/document vectors for classification."""
import random
from typing import Mapping, Optional, Sequence
import nltk
import numpy as np
from numpy.typing import NDArray


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


def generate_synthetic_data(num_sentences, num_words_per_sentence, p, vocabulary_map):
    inverted_vocabulary_map = {v: k for k, v in vocabulary_map.items()}
    documents = []
    for i in range(num_sentences):
        sentence = [
            inverted_vocabulary_map[np.random.choice(len(p), p=p)]
            for _ in range(num_words_per_sentence)
        ]
        documents.append(sentence)
    return documents


#
h0_documents = nltk.corpus.gutenberg.sents("shakespeare-hamlet.txt")
h1_documents = nltk.corpus.gutenberg.sents("bible-kjv.txt")

vocabulary_map = build_vocabulary(h0_documents, h1_documents)
X_train, y_train, X_test, y_test = generate_data_token_counts(
    h0_documents, h1_documents
)


# Train Naive Bayes
ph0, ph1, p0, p1 = train_naive_bayes(X_train, y_train)
# ph0: Probability of class 0
# ph1: Probability of class 1
# p0: Matrix of probabilities for each token in the vocabulary given class 0
# p1: Matrix of probabilities for each token in the vocabulary given class 1

# Test Naive Bayes
print("Naive Bayes (train):", test_naive_bayes(X_train, y_train, ph0, ph1, p0, p1))
print("Naive Bayes (test):", test_naive_bayes(X_test, y_test, ph0, ph1, p0, p1))

# Create synthetic data:
num_sentences = 15
num_words_per_sentence = 7

documents = generate_synthetic_data(
    num_sentences, num_words_per_sentence, p0, vocabulary_map
)
for sentence in documents:
    print(sentence)
