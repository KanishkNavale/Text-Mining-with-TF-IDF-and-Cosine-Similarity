from typing import Tuple, List
import json
import os

import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix

from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

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


def encode_labels(labels: List[str]) -> List[int]:
    """Coverts a list of binary classes to a list of bool classes

    Args:
        labels (List[str]): List of binary classes in string format.

    Returns:
        List[bool]: List of classes in boolean format.
    """
    encoded_labels: List[int] = []

    for label in tqdm(labels, desc="Encoding Labels", total=len(labels)):
        if label == 'gut':
            encoded_labels.append(1)
        else:
            encoded_labels.append(0)

    return encoded_labels


def generate_normalized_tokens_from_text(text_list: List[str]) -> List[List[str]]:
    """Convert a list of strings to list of list of normalized tokens

    Args:
        text_list (List[str]): list of strings

    Returns:
        List[List[str]]: list of list of normalized tokens
    """
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
    """Computes and returns a list of unique tokens

    Args:
        data (List[List[str]]): list of list of normalized tokens

    Returns:
        List[str]: list of unique tokens
    """
    unique_tokens_list: List[str] = []

    for text in tqdm(data, desc="Computing Unique Tokens", total=len(data)):
        unique_tokens_list.extend(text)

    return sorted(list(set(unique_tokens_list)))


def convert_list_of_tokens_to_vectors(data: List[str], dictionary: List[str]) -> csr_matrix:
    """converts a list of unique tokens to matrix of vector as features

    Args:
        data (List[str]): list of strings
        dictionary (List[str]): list of unique tokens

    Returns:
        csr_matrix: matrix of features
    """
    vectorizer = TfidfVectorizer(vocabulary=dictionary)
    return vectorizer.fit_transform(data)


def convert_csr_matrix_to_torch_tensor(data: csr_matrix, device: torch.device) -> torch.sparse_coo_tensor:
    """Converts a sparse csr_matrix to torch.tensor

    Args:
        data (csr_matrix): sparse csr_matrix
        device (torch.device): torch.device

    Returns:
        torch.sparse_coo_tensor: sparse torch tensor
    """
    data = data.tocoo()

    indices_as_tensor = torch.stack((torch.tensor(data.row), torch.tensor(data.col)))
    values_as_tensor = torch.tensor(data.data)

    return torch.sparse_coo_tensor(indices_as_tensor, values_as_tensor, dtype=torch.float32, device=device)


def optimize(features: csr_matrix,
             labels: List[bool],
             epochs: int = 10,
             learning_rate: float = 1e-3,
             penalize: bool = False) -> np.ndarray:
    """Computes optimized weights for classification

    Args:
        features (csr_matrix): matrix of features
        labels (List[bool]): list of classes
        epochs (int, optional): total number of optimization epochs. Defaults to 10.
        learning_rate (float, optional): learning rate for optimizer. Defaults to 1e-3.
        penalize (bool, optional): use regularizer or not. Defaults to False.

    Returns:
        np.ndarray: optimized weights
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    features = convert_csr_matrix_to_torch_tensor(features, device)
    labels = torch.as_tensor(labels, device=device, dtype=torch.float32)

    weights = np.random.uniform(-1.0, 1.0, (1, features.shape[1]))
    weights = torch.tensor(weights, dtype=torch.float32, device=device, requires_grad=True)

    if penalize:
        optimizer = torch.optim.Adam([weights], lr=learning_rate, weight_decay=1e-6)
    else:
        optimizer = torch.optim.Adam([weights], lr=learning_rate)

    loss_function = torch.nn.HuberLoss()

    with tqdm(range(epochs), desc='Optimizing Weights', total=epochs) as progress_bar:
        for _ in progress_bar:
            train_prediction = torch.sigmoid(torch.sparse.mm(features, weights.T))
            train_prediction = train_prediction.squeeze(dim=-1)
            training_loss = loss_function(train_prediction, labels)
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

            progress_bar.set_postfix(Loss=training_loss.item())

    return weights.squeeze(dim=-1).detach().cpu().numpy()


def predict_labels(features: csr_matrix, weights: np.ndarray) -> np.ndarray:
    """Compute predictions for given data and weights.

    Args:
        features (csr_matrix): matrix of features.
        weights (np.array): weights for computation.

    Returns:
        np.array: predicted labels.
    """
    z = np.ravel(features @ weights.T)
    a = np.where(z > 0.5, 1.0, 0.0)
    return np.array(a, dtype=np.float64)


def compute_accuracy(prediction: np.array, groundtruth: np.array) -> float:
    """Computes accuracy from predicted lables and groundtruth.

    Args:
        prediction (np.array): prediction labels.
        groundtruth (np.array): truth labels.

    Returns:
        float: accuracy 
    """
    return len(np.where(prediction == groundtruth)[0]) / len(groundtruth)


def plot_and_save_confusion_matrix(prediction: np.ndarray, groundtruth: np.ndarray, title: str = '') -> None:
    """Plots and saves the confusion matrix

    Args:
        prediction (np.array): prediction labels
        groundtruth (np.array): groundtruth labels.
        title (str, optional): Title for the plot and filename. Defaults to ''.
    """
    matrix = np.matrix(confusion_matrix(groundtruth, prediction))

    plt.figure()

    fig, ax = plt.subplots(1, 1)

    ax.matshow(matrix, cmap='GnBu')

    for x in (0, 1):
        for y in (0, 1):
            ax.text(x, y, matrix[y, x])

    ax.set_xlabel('predicted label')
    ax.set_ylabel('true label')
    ax.set_xticklabels(['', 'gut', 'schlecht'])
    ax.set_yticklabels(['', 'gut', 'schlecht'])
    ax.set_title(title)

    fig.savefig(os.path.join('torch_implementation/data', title + '.png'))


def plot_and_save_weight_histogram(weights: np.ndarray, title: str = '') -> None:
    """Plots and saves weight hsitogram

    Args:
        weights (np.array): weights
        title (str, optional): Title for the plot and filename. Defaults to ''.
    """
    plt.figure()
    plt.hist(np.ravel(weights), bins="auto")
    plt.grid(True)
    plt.title(title)
    plt.savefig(os.path.join('torch_implementation/data', title + '.png'))


def save_terms_with_their_weights(weights: np.ndarray, dictionary: np.ndarray, title: str = '') -> None:
    """Saves Terms with it's corresponding weights.

    Args:
        weights (np.ndarray): weights
        dictionary (np.ndarray): list of unique tokens
        title (str, optional): filename. Defaults to ''.
    """

    weights = np.ravel(weights)
    ranks = weights.argsort()[::-1]

    term: List[str] = []
    weight: List[float] = []

    for index in ranks:
        term.append(dictionary[index])
        weight.append(np.around(weights[index], 4))

    df = pd.DataFrame(list(zip(*[term, weight])), columns=['tokens', 'weight'])
    df.to_csv(os.path.join('torch_implementation/data', title + '.csv'), index=False)


if __name__ == "__main__":

    # Read dataset from the .csv file for Training the Perceptron
    df = pd.read_csv('datasets/games-train.csv', delimiter='\t', header=None)
    df = pd.DataFrame(df)

    labels = list(df.iloc[:, 1])
    texts = list(df.iloc[:, 3])

    texts, labels = delete_nan(texts, labels)
    encoded_labels = encode_labels(labels)
    tokanized_texts = generate_normalized_tokens_from_text(texts)
    dictionary = generate_list_of_unique_tokens(tokanized_texts)

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, encoded_labels)

    train_features = convert_list_of_tokens_to_vectors(train_texts, dictionary)
    val_features = convert_list_of_tokens_to_vectors(val_texts, dictionary)
    val_labels = np.array(val_labels, dtype=np.float64)

    optimized_weights = optimize(train_features, train_labels, epochs=20000)
    predicted_labels = predict_labels(val_features, optimized_weights)

    regularized_weights = optimize(train_features, train_labels, epochs=20000, penalize=True)
    regularized_labels = predict_labels(val_features, regularized_weights)

    data = {
        "Non Penalized Accuracy": compute_accuracy(predicted_labels, val_labels),
        "Penalized Accuracy": compute_accuracy(regularized_labels, val_labels)
    }

    # Dump .json and other data
    with open(os.path.join(os.path.abspath('torch_implementation/data'), 'inference.json'), 'w', encoding='utf8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    plot_and_save_confusion_matrix(predicted_labels, val_labels, "Non Penalized Prediction")
    plot_and_save_confusion_matrix(regularized_labels, val_labels, "Penalized Prediction")

    plot_and_save_weight_histogram(optimized_weights, "Non Penalized Weights")
    plot_and_save_weight_histogram(regularized_weights, "Penalized Weights")

    save_terms_with_their_weights(optimized_weights, dictionary, "Non Penalized Terms")
    save_terms_with_their_weights(regularized_weights, dictionary, "Penalized Terms")
