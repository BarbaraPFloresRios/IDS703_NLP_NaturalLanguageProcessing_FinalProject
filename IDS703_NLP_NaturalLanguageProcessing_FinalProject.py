# Barbara Flores
# Daniela Jimenez
# Shaila Guereca


import random
from typing import Mapping, Optional, Sequence
import nltk
import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from tabulate import tabulate


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
    vocabulary_set = set(
        token for sentence in h0_documents + h1_documents for token in sentence
    )
    vocabulary_set.discard(None)
    vocabulary = sorted(vocabulary_set) + [None]
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
    predictions = []
    misclassified = []

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

        predictions.append(guess)

        if guess != label:
            misclassified.append(sentence)

        else:
            num_correct += 1

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Create confusion matrix table
    table = [
        ["", "Predicted 0", "Predicted 1"],
        ["Actual 0", cm[0][0], cm[0][1]],
        ["Actual 1", cm[1][0], cm[1][1]],
    ]

    accuracy = num_correct / len(y_test)

    return accuracy, table, misclassified


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
    num_correct = 0
    predictions = []
    misclassified = []

    # Transform X_test using TF-IDF
    X_test_tfidf = tfidf.transform(X_test)

    # Predict using the trained logistic regression model
    predictions = clf.predict(X_test_tfidf)

    for i in range(len(y_test)):
        sentence = X_test[i]
        label = y_test[i]
        guess = predictions[i]

        if guess != label:
            misclassified.append(sentence)

        else:
            num_correct += 1

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Create confusion matrix table
    table = [
        ["", "Predicted 0", "Predicted 1"],
        ["Actual 0", cm[0][0], cm[0][1]],
        ["Actual 1", cm[1][0], cm[1][1]],
    ]

    accuracy = num_correct / len(y_test)

    return accuracy, table, misclassified


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


random.seed(47)
## RESULTS
# Load Data
h0_documents = nltk.corpus.gutenberg.sents("shakespeare-hamlet.txt")
h1_documents = nltk.corpus.gutenberg.sents("bible-kjv.txt")[:3106]
h0_documents = h0_documents[: max(len(h0_documents), len(h1_documents))]
h1_documents = h1_documents[: max(len(h0_documents), len(h1_documents))]

# Print information about the documents
print("DOCUMENT INFORMATION\n")
print(f"h0 = Hamlet by Shakespeare")
print(f"Number of sentences in h0: {len(h0_documents):,}")
print(
    f"Number of unique words in h0: {len(set(token for sentence in h0_documents for token in sentence)):,}"
)
print()

print(f"h1 = Bible")
print(f"Number of sentences in h1: {len(h1_documents):,}")
print(
    f"Number of unique words in h1: {len(set(token for sentence in h1_documents for token in sentence)):,}"
)
print()

vocabulary_map = build_vocabulary(h0_documents, h1_documents)
X_train, y_train, X_test, y_test = generate_data_token_counts(
    h0_documents, h1_documents
)

print("_" * 50)
# Train Naive Bayes
ph0, ph1, p0, p1 = train_naive_bayes(X_train, y_train)

# Test Naive Bayes
print("\nRESULTS NAIVE BAYES\n")

# Test Naive Bayes on training set
(
    accuracy_train_nb,
    confusion_matrix_train_nb,
    misclassified_sentences_train_nb,
) = test_naive_bayes(X_train, y_train, ph0, ph1, p0, p1)

# Test Naive Bayes on testing set
(
    accuracy_test_nb,
    confusion_matrix_test_nb,
    misclassified_sentences_test_nb,
) = test_naive_bayes(X_test, y_test, ph0, ph1, p0, p1)

# Accuracy
print("Naive Bayes (train):", accuracy_train_nb)
print("Naive Bayes (test):", accuracy_test_nb)

# Print confusion matrix for Naive Bayes (Train)
print("\nConfusion Matrix for Naive Bayes (Train)")
print(tabulate(confusion_matrix_train_nb, headers="firstrow", tablefmt="fancy_grid"))


# Print confusion matrix using tabulate
print("\nConfusion Matrix for Naive Bayes (Test)")
print(tabulate(confusion_matrix_test_nb, headers="firstrow", tablefmt="fancy_grid"))


print("\nMisclassified Sentences for Naive Bayes (Test)\n")
inverted_vocabulary_map = {v: k for k, v in vocabulary_map.items()}
for sentence_embedding in misclassified_sentences_test_nb:
    # Convert sentence embedding back to tokens
    sentence_tokens = [
        inverted_vocabulary_map[idx]
        for idx, value in enumerate(sentence_embedding)
        if value == 1
    ]
    print(f"- {' '.join(sentence_tokens)}")


print("_" * 50)
print("\n\nRESULTS LOGISTIC REGRESSION WITH TF-IDF\n")

# Train Logistic Regression with TF-IDF
clf_logistic, tfidf_logistic, X_train_tfidf = train_logistic_regression_with_tfidf(
    X_train, y_train
)

# Test Logistic Regression with TF-IDF
(
    accuracy_train_lg,
    confusion_matrix_train_lg,
    misclassified_sentences_train_lg,
) = test_logistic_regression_with_tfidf(X_train, y_train, clf_logistic, tfidf_logistic)
(
    accuracy_test_lg,
    confusion_matrix_test_lg,
    misclassified_sentences_test_lg,
) = test_logistic_regression_with_tfidf(X_test, y_test, clf_logistic, tfidf_logistic)

print("Logistic Regression with TF-IDF (train):", accuracy_train_lg)
print("Logistic Regression with TF-IDF (test):", accuracy_test_lg)


# Print confusion matrix for Logistic Regression with TF-IDF (Train)
print("\nConfusion Matrix for Logistic Regression with TF-IDF (Train)")
print(tabulate(confusion_matrix_train_lg, headers="firstrow", tablefmt="fancy_grid"))


# Print confusion matrix using tabulate
print("\nConfusion Matrix for Logistic Regression with TF-IDF (Test)")
print(tabulate(confusion_matrix_test_lg, headers="firstrow", tablefmt="fancy_grid"))

print("\nMisclassified Sentences for Logistic Regression with TF-IDF (Test)\n")
inverted_vocabulary_map = {v: k for k, v in vocabulary_map.items()}
for sentence_embedding in misclassified_sentences_test_lg:
    # Convert sentence embedding back to tokens
    sentence_tokens = [
        inverted_vocabulary_map[idx]
        for idx, value in enumerate(sentence_embedding)
        if value == 1
    ]
    print(f"- {' '.join(sentence_tokens)}")


# Generate Sintetic Data
num_sentences = 1000
num_words_per_sentence = 15

# h0: Hamlet  h1: Bible
synthetic_data_h0 = generate_synthetic_data(
    num_sentences, num_words_per_sentence, p0, vocabulary_map
)
synthetic_data_h1 = generate_synthetic_data(
    num_sentences, num_words_per_sentence, p1, vocabulary_map
)


vocabulary_map_synthetic_data = build_vocabulary(synthetic_data_h0, synthetic_data_h1)
(
    X_train_synthetic_dat,
    y_train_synthetic_dat,
    X_test_synthetic_dat,
    y_test_synthetic_dat,
) = generate_data_token_counts(synthetic_data_h0, synthetic_data_h1)


# Display information about synthetic data
print("_" * 50)
print("\n")
print("SYNTHETIC DATA INFORMATION\n")

print(f"h0 = Synthetic Hamlet by Shakespeare")
print(f"Number of sentences in synthetic data h0: {len(synthetic_data_h0):,}")
print(
    f"Number of unique words in synthetic data h0: {len(set(token for sentence in synthetic_data_h0 for token in sentence)):,}"
)
print()

print(f"h1 = Synthetic Bible")
print(f"Number of sentences in synthetic data h1: {len(synthetic_data_h1):,}")
print(
    f"Number of unique words in synthetic data h1: {len(set(token for sentence in synthetic_data_h1 for token in sentence)):,}"
)
print()

print("_" * 50)
print("\nSYNTHETIC RESULTS\n")


# Train Naive Bayes on Synthetic Data
ph0_synthetic, ph1_synthetic, p0_synthetic, p1_synthetic = train_naive_bayes(
    X_train_synthetic_dat, y_train_synthetic_dat
)

# Test Naive Bayes on Synthetic Data
print("\nRESULTS NAIVE BAYES ON SYNTHETIC DATA\n")

# Test Naive Bayes on training set (synthetic)
(
    accuracy_train_nb_synthetic,
    confusion_matrix_train_nb_synthetic,
    misclassified_sentences_train_nb_synthetic,
) = test_naive_bayes(
    X_train_synthetic_dat,
    y_train_synthetic_dat,
    ph0_synthetic,
    ph1_synthetic,
    p0_synthetic,
    p1_synthetic,
)

# Test Naive Bayes on testing set (synthetic)
(
    accuracy_test_nb_synthetic,
    confusion_matrix_test_nb_synthetic,
    misclassified_sentences_test_nb_synthetic,
) = test_naive_bayes(
    X_test_synthetic_dat,
    y_test_synthetic_dat,
    ph0_synthetic,
    ph1_synthetic,
    p0_synthetic,
    p1_synthetic,
)

# Accuracy
print("Naive Bayes on Synthetic Data (train):", accuracy_train_nb_synthetic)
print("Naive Bayes on Synthetic Data (test):", accuracy_test_nb_synthetic)

# Print confusion matrix for Naive Bayes on Synthetic Data (Train)
print("\nConfusion Matrix for Naive Bayes on Synthetic Data (Train)")
print(
    tabulate(
        confusion_matrix_train_nb_synthetic, headers="firstrow", tablefmt="fancy_grid"
    )
)

# Print confusion matrix for Naive Bayes on Synthetic Data (Test)
print("\nConfusion Matrix for Naive Bayes on Synthetic Data (Test)")
print(
    tabulate(
        confusion_matrix_test_nb_synthetic, headers="firstrow", tablefmt="fancy_grid"
    )
)

print("\nMisclassified Sentences for Naive Bayes on Synthetic Data (Test)\n")
for sentence_embedding in misclassified_sentences_test_nb_synthetic:
    # Convert sentence embedding back to tokens
    sentence_tokens = [
        inverted_vocabulary_map[idx]
        for idx, value in enumerate(sentence_embedding)
        if value == 1
    ]
    print(f"- {' '.join(sentence_tokens)}")

# Train Logistic Regression with TF-IDF on Synthetic Data
(
    clf_logistic_synthetic,
    tfidf_logistic_synthetic,
    X_train_tfidf_synthetic,
) = train_logistic_regression_with_tfidf(X_train_synthetic_dat, y_train_synthetic_dat)

# Test Logistic Regression with TF-IDF on Synthetic Data
(
    accuracy_train_lg_synthetic,
    confusion_matrix_train_lg_synthetic,
    misclassified_sentences_train_lg_synthetic,
) = test_logistic_regression_with_tfidf(
    X_train_synthetic_dat,
    y_train_synthetic_dat,
    clf_logistic_synthetic,
    tfidf_logistic_synthetic,
)

(
    accuracy_test_lg_synthetic,
    confusion_matrix_test_lg_synthetic,
    misclassified_sentences_test_lg_synthetic,
) = test_logistic_regression_with_tfidf(
    X_test_synthetic_dat,
    y_test_synthetic_dat,
    clf_logistic_synthetic,
    tfidf_logistic_synthetic,
)
print("_" * 50)
print("\nRESULTS LOGISTIC REGRESSION WITH TF-IDF ON SYNTHETIC DATA\n")

# Accuracy
print(
    "Logistic Regression with TF-IDF on Synthetic Data (train):",
    accuracy_train_lg_synthetic,
)
print(
    "Logistic Regression with TF-IDF on Synthetic Data (test):",
    accuracy_test_lg_synthetic,
)

# Print confusion matrix for Logistic Regression with TF-IDF on Synthetic Data (Train)
print(
    "\nConfusion Matrix for Logistic Regression with TF-IDF on Synthetic Data (Train)"
)
print(
    tabulate(
        confusion_matrix_train_lg_synthetic, headers="firstrow", tablefmt="fancy_grid"
    )
)

# Print confusion matrix for Logistic Regression with TF-IDF on Synthetic Data (Test)
print("\nConfusion Matrix for Logistic Regression with TF-IDF on Synthetic Data (Test)")
print(
    tabulate(
        confusion_matrix_test_lg_synthetic, headers="firstrow", tablefmt="fancy_grid"
    )
)

print(
    "\nMisclassified Sentences for Logistic Regression with TF-IDF on Synthetic Data (Test)\n"
)
for sentence_embedding in misclassified_sentences_test_lg_synthetic:
    # Convert sentence embedding back to tokens
    sentence_tokens = [
        inverted_vocabulary_map[idx]
        for idx, value in enumerate(sentence_embedding)
        if value == 1
    ]
    print(f"- {' '.join(sentence_tokens)}")
