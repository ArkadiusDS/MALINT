import os
import pandas as pd
from transformers import TrainerCallback


class SaveMetricsCallback(TrainerCallback):

    def __init__(self, csv_file_name, hyperparameters):
        """
        Initializes the SaveMetricsCallback class.

        Parameters:
        - csv_file_name (str): The name of the CSV file to save metrics.
        - hyperparameters (dict): Dictionary containing hyperparameters.
        """
        super().__init__()
        self.df = pd.DataFrame()
        self.file_name = csv_file_name

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.file_name), exist_ok=True)

        self.df_hyperparameters = pd.DataFrame([hyperparameters])

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        Event called after an evaluation phase.
        Appends evaluated metrics to the DataFrame.
        """
        self.df = pd.concat([self.df, pd.DataFrame([metrics])])

    def on_train_end(self, args, state, control, **kwargs):
        """
        Event called at the end of training.
        Concatenates hyperparameters DataFrame with metrics DataFrame and saves to CSV file.
        """
        self.df = pd.concat([self.df, self.df_hyperparameters], axis=1)
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(self.file_name), exist_ok=True)
        self.df.to_csv(self.file_name, index=False)