import json
import os
import torch
import ast
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
)
from utils.utils import (
    compute_multi_label_metrics, load_config
)


def predict_labels(text, tokenizer, model, id2label):
    """
    Function that predicts labels for development dataset
    """
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    encoding = {k: v.to(model.device) for k, v in encoding.items()}

    outputs = model(**encoding)

    logits = outputs.logits

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.30)] = 1

    # df_pred = pd.DataFrame(predictions, columns=LABELS)
    # turn predicted id's into actual label names
    predicted_labels = ",".join(id2label[idx] for idx, label in enumerate(predictions) if label == 1.0)
    return predicted_labels


LABELS = ['CPV', 'PSSA', 'UIOA', 'PASV', 'UCPI']

id2label = {idx: label for idx, label in enumerate(LABELS)}
label2id = {label: idx for idx, label in enumerate(LABELS)}


def labels_one_hot_encoding(df, labels_col_name):
    """
    Returns DataFrame with one-hot encoded label columns.
    """
    df = df.copy()
    df[labels_col_name] = df[labels_col_name].apply(ast.literal_eval)

    for label in LABELS:
        df[label] = df[labels_col_name].apply(lambda x: label in x)

    return df.drop(columns=[labels_col_name])


if __name__ == '__main__':

    config = load_config("config/config.yaml")
    test_data = pd.read_csv(config["intention"]["data"]["test"], encoding='utf-8')

    for experiment in config["intention"]["models"]:

        model_saved_path = experiment["path_to_save_model"]

        if "v3" in experiment["model"].lower():
            tokenizer = AutoTokenizer.from_pretrained(experiment["model"], use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(experiment["model"])

        model = AutoModelForSequenceClassification.from_pretrained(model_saved_path)

        test_ground_truth = test_data.copy()
        test_data["pred"] = test_data['content'].apply(
            lambda x: predict_labels(text=x, tokenizer=tokenizer, model=model, id2label=id2label)
        )
        test_data.to_csv(model_saved_path.split("/")[-1] + "pred_encoded.csv", index=False)

        for label in LABELS:
            test_data[label] = test_data["pred"].apply(lambda x: label in x)

        evaluation_results = compute_multi_label_metrics(
            test_ground_truth[LABELS],
            test_data[LABELS],
            target_names=LABELS
        )
        # Save evaluation metrics to a JSON file

        output_file_path = experiment["test_metrics"] + ".json"
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as output_file:
            json.dump(evaluation_results, output_file, indent=4)
