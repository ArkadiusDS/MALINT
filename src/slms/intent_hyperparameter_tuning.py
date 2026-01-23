import os
import random
import transformers
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import EvalPrediction
import torch
from utils.custom_callbacks import SaveMetricsCallback
from utils.utils import preprocess_data, load_config, cartesian_product


def multi_label_metrics(predictions, labels, target_names, threshold=0.30):
    """
    Function with last layer for multilabel classification and computing metrics
    """
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    clf_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_macro_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1_micro': f1_micro_average,
               'f1_macro': f1_macro_average,
               'f1_macro_weighted': f1_macro_weighted,
               'accuracy': accuracy,
               'classification_report': clf_report}
    return metrics


#
def compute_metrics(p: EvalPrediction):
    """
    Function for computing metrics
    """
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        target_names=LABELS,
        labels=p.label_ids)
    return result


if __name__ == "__main__":

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(42)

    config = load_config()

    train_data = pd.read_csv(config["intention"]["data"]["train"], encoding='utf-8')
    validation_data = pd.read_csv(config["intention"]["data"]["validation"], encoding='utf-8')

    LABELS = ['CPV', 'PSSA', 'UIOA', 'PASV', 'UCPI']

    id2label = {idx: label for idx, label in enumerate(LABELS)}
    label2id = {label: idx for idx, label in enumerate(LABELS)}

    # Assuming you already have `train` and `valid` as pandas DataFrames
    train_dataset = Dataset.from_pandas(train_data, preserve_index=False)
    valid_dataset = Dataset.from_pandas(validation_data, preserve_index=False)

    # Optionally combine into a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset
    })

    for experiment in config["intention"]["models"]:

        tokenizer = AutoTokenizer.from_pretrained(experiment["model"], use_fast=False)

        # Apply preprocessing (e.g. tokenization)
        train_dataset = train_dataset.map(
            lambda batch: preprocess_data(batch, tokenizer, LABELS),
            batched=True,
            remove_columns=train_dataset.column_names
        )

        valid_dataset = valid_dataset.map(
            lambda batch: preprocess_data(batch, tokenizer, LABELS),
            batched=True,
            remove_columns=valid_dataset.column_names
        )
        # encoded_dataset.set_format("torch")
        model = AutoModelForSequenceClassification.from_pretrained(experiment["model"],
                                                                   problem_type="multi_label_classification",
                                                                   num_labels=len(LABELS),
                                                                   id2label=id2label,
                                                                   label2id=label2id)
        # These settings and code was used for tuning hyperparameters
        hyper_parameters_dict = {
            "eval_strategy": ["steps"],
            "learning_rate": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
            "per_device_train_batch_size": [4],
            "per_device_eval_batch_size": [4],
            "num_train_epochs": [5],
            "warmup_ratio": [0.06, 0.1],
            "weight_decay": [0.01, 0.03, 0.05, 0.1],
            "fp16": [True],
            "metric_for_best_model": ["f1_macro_weighted"],
            "load_best_model_at_end": [True],
            "save_total_limit": [2],
            "greater_is_better": [True],
            "save_strategy": ["steps"],
            "eval_steps": [50]
        }

        set_of_hyper_parameters = cartesian_product(hyper_parameters_dict)

        for ind, hyperparameters in enumerate(set_of_hyper_parameters):
            args = TrainingArguments(
                output_dir=experiment["output"] + "_" + str(ind),
                eval_strategy=hyperparameters["eval_strategy"],
                learning_rate=hyperparameters["learning_rate"],
                num_train_epochs=hyperparameters["num_train_epochs"],
                warmup_ratio=hyperparameters["warmup_ratio"],
                weight_decay=hyperparameters["weight_decay"],
                fp16=hyperparameters["fp16"],
                metric_for_best_model=hyperparameters["metric_for_best_model"],
                eval_steps=hyperparameters["eval_steps"],
                load_best_model_at_end=hyperparameters["load_best_model_at_end"],
                save_total_limit=hyperparameters["save_total_limit"],
                greater_is_better=hyperparameters["greater_is_better"],
                save_strategy=hyperparameters["save_strategy"],
            )

            trainer = Trainer(
                model,
                args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[SaveMetricsCallback(
                    csv_file_name=experiment["valid_metrics"] + "_" + str(ind) + ".csv",
                    hyperparameters=hyperparameters)
                ]
            )

            trainer.train()
            # model_saved_path = experiment["path_to_save_model"] + "_" + str(ind)
            # # Ensure the directory exists
            # os.makedirs(model_saved_path, exist_ok=True)
            #
            # trainer.save_model(model_saved_path)
