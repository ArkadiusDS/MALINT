import itertools
import yaml
import torch
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score


# Create a PyTorch Dataset for the training and validation sets
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_config(file_path='config.yaml'):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


def preprocess_data(examples, tokenizer, labels):
    """
    Tokenizing and preprocessing data for model training
    """
    # take a batch of texts
    text = examples["content"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding


def cartesian_product(hyperparameters):
    """ Returns cartesian product of all hyperparameters given in dictionary of lists"""
    return (dict(zip(hyperparameters.keys(), values)) for values in itertools.product(*hyperparameters.values()))


def compute_metrics_for_test_data(y_true, y_pred):
    clf_report = classification_report(y_true, y_pred, output_dict=True)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_macro_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {
        'f1': f1,
        'f1_micro': f1_micro_average,
        'f1_macro': f1_macro_average,
        'f1_macro_weighted': f1_macro_weighted,
        'accuracy': accuracy,
        'classification_report': clf_report
    }

    return metrics


def predict_intent(text, tokenizer, model):
    """
    Function that predicts the label for input text using argmax
    """

    tokenized_text = tokenizer([text], truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokenized_text)

    logits = outputs.logits
    probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    predicted_label = np.argmax(probabilities)
    return predicted_label



def save_data_to_json(df, file_name):
    """
    Function saves dataset into json file
    """
    df.to_json(file_name, lines=True, orient="records")


def compute_multi_label_metrics(y_true, y_pred, target_names):
    """
    Function with last layer for multilabel classification and computing metrics
    """
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    clf_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_macro_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1_micro': f1_micro_average,
               'f1_macro': f1_macro_average,
               'f1_macro_weighted':f1_macro_weighted,
               'accuracy': accuracy,
               'classification_report': clf_report}
    return metrics

def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    y_pred = pred.predictions.argmax(-1)
    f1 = f1_score(labels, y_pred)
    f1_micro_average = f1_score(y_true=labels, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=labels, y_pred=y_pred, average='macro')
    f1_macro_weighted = f1_score(y_true=labels, y_pred=y_pred, average='weighted')
    acc = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    return {
        'f1': f1,
        'f1_micro': f1_micro_average,
        'f1_macro': f1_macro_average,
        'f1_macro_weighted': f1_macro_weighted,
        'accuracy': acc,
        'precision': precision,
        'recall': recall
    }
