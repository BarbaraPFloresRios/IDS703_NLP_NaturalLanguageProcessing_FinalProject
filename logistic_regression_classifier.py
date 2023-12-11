"""Demonstrate Naive Bayes classification."""
import random
from typing import List, Mapping, Optional, Sequence

import nltk
import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

FloatArray = NDArray[np.float64]

random.seed(43)
nltk.data.path.append("/Users/shailaguereca/lib/nltk_data")
hamlet = nltk.corpus.gutenberg.sents("shakespeare-hamlet.txt")  # Austen
bible = nltk.corpus.gutenberg.sents("bible-kjv.txt")  # Carroll

# Use only the first 3106 sentences of the bible
bible = bible[:3106]

# Length of each document
print(len(hamlet))
print(len(bible))

vocabulary = sorted(set(token for sentence in hamlet + bible for token in sentence)) + [
    None
]
vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}

# print(vocabulary)


def onehot(
    vocabulary_map: Mapping[Optional[str], int], token: Optional[str]
) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary_map),))
    idx = vocabulary_map.get(token, len(vocabulary_map) - 1)
    embedding[idx] = 1
    return embedding


def generate_document_embedding(
    token_embeddings: Sequence[FloatArray],
) -> FloatArray:
    """Sum the token embeddings."""
    return np.array(token_embeddings).sum(axis=0)  # / len(token_embeddings)


# assemble training and testing data
h0_observations = [
    (
        generate_document_embedding(
            [onehot(vocabulary_map, token) for token in sentence]
        ),
        0,
    )
    for sentence in hamlet
]
h1_observations = [
    (
        generate_document_embedding(
            [onehot(vocabulary_map, token) for token in sentence]
        ),
        1,
    )
    for sentence in bible
]

all_data = h0_observations + h1_observations

random.shuffle(all_data)
test_percent = 10
break_idx = round(test_percent / 100 * len(all_data))

training_data = all_data[break_idx:]
testing_data = all_data[:break_idx]

X_train = [observation[0] for observation in training_data]
y_train = [observation[1] for observation in training_data]
X_test = [observation[0] for observation in testing_data]
y_test = [observation[1] for observation in testing_data]

# train logistic regression with TF-IDF
tfidf = TfidfTransformer(norm=None).fit(X_train)
X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)

print("num samples:, num features:", X_train.shape)

clf = LogisticRegression(random_state=0, max_iter=3000, solver="sag").fit(
    X_train, y_train
)
print("tf-idf:", clf.score(X_train, y_train))
