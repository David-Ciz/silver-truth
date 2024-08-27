import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_classification_metrics(phase, labels, predictions) -> dict:
    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions)
    precision = precision_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    # format to dictionary for brevity
    metrics = {f'{phase}_accuracy': accuracy,
               f'{phase}_recall': recall,
               f'{phase}_precision': precision,
               f'{phase}_f1': f1,
               f'{phase}_TP': tp,
               f'{phase}_TN': tn,
               f'{phase}_FP': fp,
               f'{phase}_FN': fn}
    return metrics


def calculate_regression_metrics(prefix, true_labels, predictions):
    mse = mean_squared_error(true_labels, predictions)
    mae = mean_absolute_error(true_labels, predictions)

    metrics = {
        f'{prefix}_mse': mse,
        f'{prefix}_mae': mae,
        f'{prefix}_rmse': np.sqrt(mse)
    }

    return metrics