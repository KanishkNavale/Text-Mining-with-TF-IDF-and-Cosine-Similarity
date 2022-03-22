from typing import Tuple, List

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix

from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def delete_nan(texts: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Deletes Nan from the list of text and removes corresponding labels.

    Args:
        texts (List[str]): List of text
        labels (List[str]): List of labels

    Returns:
        Tuple[List[str], List[str]]: Tuple of 'NaN' free text and corresponding labels.
    """
    filtered_texts: List[str] = []
    filtered_labels: List[str] = []

    for index, text in tqdm(enumerate(texts), desc="Deleting 'NaN' from data", total=len(texts)):
        if not pd.isna(text):
            filtered_texts.append(texts[index])
            filtered_labels.append(labels[index])

    return filtered_texts, filtered_labels


def encode_labels(labels: List[str]) -> List[bool]:
    """Coverts a list of binary classes to a list of bool classes

    Args:
        labels (List[str]): List of binary classes in string format.

    Returns:
        List[bool]: List of classes in boolean format.
    """
    encoded_labels: List[bool] = []

    for label in tqdm(labels, desc="Encoding Labels", total=len(labels)):
        if label == 'gut':
            encoded_labels.append(True)
        else:
            encoded_labels.append(False)

    return encoded_labels


def generate_normalized_tokens_from_text(text_list: List[str]) -> List[List[str]]:
    normalized_tokens_from_text: List[List[str]] = []

    for text in tqdm(text_list, desc='Converting text to tokens', total=len(text_list)):
        tokens = word_tokenize(text, language="german")
        tokens = [word.lower() for word in tokens if word.isalpha()]

        lemma = WordNetLemmatizer()
        tokens = [lemma.lemmatize(word, pos="v") for word in tokens]
        tokens = [lemma.lemmatize(word, pos="n") for word in tokens]

        normalized_tokens_from_text.append(tokens)

    return normalized_tokens_from_text


def generate_list_of_unique_tokens(data: List[List[str]]) -> List[str]:
    unique_tokens_list: List[str] = []

    for text in tqdm(data, desc="Computing Unique Tokens", total=len(data)):
        unique_tokens_list.extend(text)

    return sorted(list(set(unique_tokens_list)))


def convert_list_of_tokens_to_vectors(data: List[List[str]], dictionary: List[str]):
    vectorizer = TfidfVectorizer(vocabulary=dictionary)
    return vectorizer.fit_transform(data)


def convert_csr_matrix_to_torch_tensor(data: csr_matrix, device: torch.device):
    data = data.tocoo()

    indices_as_tensor = torch.stack((torch.tensor(data.row), torch.tensor(data.col)))
    values_as_tensor = torch.tensor(data.data)

    return torch.sparse_coo_tensor(indices_as_tensor, values_as_tensor, dtype=torch.float32, device=device)


def optimize(features: csr_matrix, labels: List[bool], epochs: int = 10, learning_rate: float = 1e-3):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weights = np.random.uniform(-1.0, 1.0, (1, features.shape[1]))
    weights = torch.tensor(weights, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([weights], lr=learning_rate)
    loss_function = torch.nn.BCELoss()

    features = convert_csr_matrix_to_torch_tensor(features, device)
    lables = torch.as_tensor(labels, device=device, dtype=torch.float32)

    with tqdm(range(epochs), desc='Optimizing Weights', total=epochs) as progress_bar:
        for epoch in progress_bar:
            train_prediction = torch.sigmoid(torch.sparse.mm(features, weights.T))
            training_loss: torch.Tensor = loss_function(train_prediction.squeeze(dim=-1), lables)
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

            progress_bar.set_postfix(Loss=training_loss.item())

    return weights


if __name__ == "__main__":

    # Read dataset from the .csv file for Training the Perceptron
    df = pd.read_csv('datasets/games-train.csv', delimiter='\t', header=None)
    df = pd.DataFrame(df)

    train_labels = list(df.iloc[:, 1])
    train_texts = list(df.iloc[:, 3])

    train_texts, train_labels = delete_nan(train_texts, train_labels)

    encoded_labels = encode_labels(train_labels)
    tokanized_texts = generate_normalized_tokens_from_text(train_texts)

    dictionary = generate_list_of_unique_tokens(tokanized_texts)
    features = convert_list_of_tokens_to_vectors(train_texts, dictionary)

    # Compute optimized weights
    optimized_weights = optimize(features, encoded_labels, epochs=10000)
