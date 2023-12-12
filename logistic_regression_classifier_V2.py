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
        " ".join(sentence),  # Keep track of the original sentence
    )
    for sentence in hamlet
]
h1_observations = [
    (
        generate_document_embedding(
            [onehot(vocabulary_map, token) for token in sentence]
        ),
        1,
        " ".join(sentence),  # Keep track of the original sentence
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
print(X_train[:1])

y_train = [observation[1] for observation in training_data]
original_sentences_train = [observation[2] for observation in training_data]
# Keep track of the original sentences

X_test = [observation[0] for observation in testing_data]
y_test = [observation[1] for observation in testing_data]
original_sentences_test = [observation[2] for observation in testing_data]
print(y_train[:1])


# train logistic regression with TF-IDF
tfidf = TfidfTransformer(norm=None).fit(X_train)
X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)

print("num samples:, num features:", X_train.shape)

clf = LogisticRegression(random_state=0, max_iter=3000, solver="sag").fit(
    X_train, y_train
)
print("tf-idf:", clf.score(X_train, y_train))


#########################################

### Confusion Matrix

from sklearn.metrics import confusion_matrix
from tabulate import tabulate

# Assume clf is your trained classifier and X_train, y_train are your training data and labels
y_pred = clf.predict(X_train)

# Create confusion matrix
cm = confusion_matrix(y_train, y_pred)

# Create a table with the confusion matrix
table = [
    ["", "Predicted 0", "Predicted 1"],
    ["Actual 0", cm[0][0], cm[0][1]],
    ["Actual 1", cm[1][0], cm[1][1]],
]

# Print the table with tabulate
print("Confusion Matrix")
print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

#####################################################


# Assume `sentences` is your original list of sentences corresponding to `X_train`
misclassified = np.where(y_train != y_pred)

# Print the misclassified sentences
for index in misclassified[0]:
    print(f"Sentence: {all_data[index]}")
    print(f"Actual label: {y_train[index]}")
    print(f"Predicted label: {y_pred[index]}\n")


# Find the misclassified sentences
misclassified_indices = np.where(y_train != y_pred)[0]
misclassified_sentences = [original_sentences_train[i] for i in misclassified_indices]

# Print the misclassified sentences
for sentence in misclassified_sentences:
    print(sentence)


#############
# word clouds can visualize the TF-IDF weights for each class

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Get feature names (vocabulary)
feature_names = np.array(vocabulary[:-1])  # Exclude the last 'None' token

# Get TF-IDF values for each feature
class_coef = clf.coef_.ravel()  # Coefficients for the class

# Create a mapping of feature names to their respective TF-IDF values
class_tfidf = {feature: coef for feature, coef in zip(feature_names, class_coef)}

# Create WordCloud object based on TF-IDF values
wordcloud_class = WordCloud(
    width=800, height=400, background_color="white"
).generate_from_frequencies(class_tfidf)

# Plot word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_class, interpolation="bilinear")
plt.title("Word Cloud")
plt.axis("off")

plt.tight_layout()
plt.show()


######################################
# TEST DATA

clf = LogisticRegression(random_state=0, max_iter=3000, solver="sag").fit(
    X_train, y_train
)

print("tf-idf:TEST", clf.score(X_test, y_test))

# Assume clf is your trained classifier and X_train, y_train are your training data and labels
y_pred2 = clf.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred2)

# Create a table with the confusion matrix
table = [
    ["", "Predicted 0", "Predicted 1"],
    ["Actual 0", cm[0][0], cm[0][1]],
    ["Actual 1", cm[1][0], cm[1][1]],
]

# Print the table with tabulate
print("Confusion Matrix")
print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

misclassified = np.where(y_test != y_pred2)

# Print the misclassified sentences
for index in misclassified[0]:
    #    print(f"Sentence: {all_data[index]}")
    print(f"Actual label: {y_test[index]}")
    print(f"Predicted label: {y_pred2[index]}\n")


# Find the misclassified sentences
misclassified_indices = np.where(y_test != y_pred2)[0]
misclassified_sentences = [original_sentences_test[i] for i in misclassified_indices]

# Print the misclassified sentences
for sentence in misclassified_sentences:
    print(sentence)
