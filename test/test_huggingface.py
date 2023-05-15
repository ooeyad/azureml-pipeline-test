import unittest
import argparse
from unittest import mock
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append("C:/Users/t6805i0/PycharmProjects/AzmlTestDev")
from huggingface.huggingface import perform_training,prepare_training_datatests, preprocess_data, multi_label_metrics


class HuggingFaceTestCase(unittest.TestCase):

    # @patch('huggingface.huggingface.get_args')
    # @mock.patch("os.listdir", return_value=["test_data.csv"])
    # @mock.patch("pandas.read_csv")
    # @mock.patch("pandas.DataFrame.to_csv")
    # def test_prepare_training_datatests(self, mock_csv_writer, mock_csv_reader, mock_get_args,mock_listdir):
    #     # Mock the return value of get_args()
    #     prepped_data_path = "../data/tmp/prepped_data"
    #     status_output_path = "../data/tmp/status_output"
    #     os.makedirs(prepped_data_path, exist_ok=True)
    #     os.makedirs(status_output_path, exist_ok=True)
    #
    #     mock_args = MagicMock()
    #     mock_args.prepped_data = prepped_data_path
    #     mock_args.status_output = status_output_path
    #     mock_get_args.return_value = mock_args
    #     mock_csv_reader.return_value = pd.DataFrame(
    #         {"col1": ["A", "B", "B", "C", "D", "E"], "col2": [1, 2, 2, 3, 4, 5]})
    #     mock_csv_writer.return_value = None
    #
    #     # Mock the behavior of pd.read_csv()
    #     mock_dataframe = pd.DataFrame({'CONCATENATED_TEXT': ['text1', 'text2', 'text3']})
    #     with patch('pandas.read_csv', return_value=mock_dataframe):
    #         dataset = prepare_training_datatests()
    #
    #     # Assert the expected output
    #     self.assertEqual(len(dataset), 3)  # Assuming 'train', 'test', and 'validation' datasets are returned
    #
    # # @patch('huggingface.huggingface.f1_score')
    # # @patch('huggingface.huggingface.roc_auc_score')
    # # @patch('huggingface.huggingface.accuracy_score')
    # def test_multi_label_metrics(self): #, mock_accuracy_score, mock_roc_auc_score, mock_f1_score):
    #     # Create sample predictions and labels
    #     predictions = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
    #     labels = [[1, 0], [0, 1], [1, 0]]
    #
    #     # mock_f1_score.return_value = 0.6666666666666666
    #     # mock_roc_auc_score.return_value = 0.6666666666666666
    #     # mock_accuracy_score.return_value = 0.7
    #
    #     # Call the multi_label_metrics function
    #     result = multi_label_metrics(predictions, labels, threshold=0.5)
    #
    #     print('actual result: ' + str(result))
    #     # Assert the expected metrics
    #     expected_metrics = {'f1': 0.6666666666666666, 'roc_auc': 0.5, 'accuracy': 0.0}
    #     self.assertEqual(result, expected_metrics)
    #     pass
    #
    # @patch('huggingface.huggingface.AutoTokenizer.from_pretrained')
    # def test_preprocess_data(self, mock_from_pretrained):
    #     # Mock the behavior of AutoTokenizer.from_pretrained()
    #     mock_tokenizer = MagicMock()
    #     mock_from_pretrained.return_value = mock_tokenizer
    #
    #     # Create a sample input example
    #     train_df = pd.DataFrame({"i":[1], "CONCATENATED_TEXT": ["Sample text train"], "a":[0],"b":[1]})
    #     test_df = pd.DataFrame({"i":[1], "CONCATENATED_TEXT": ["Sample text test"], "a":[1],"b":[0]})
    #     train_df.set_index("i")
    #     test_df.set_index("i")
    #     example = {"train":train_df ,
    #                "test":test_df
    #                }
    #
    #
    #
    #     # Set up the expected output
    #     expected_output = {
    #         "input_ids": [1, 2, 3, 4, 5],
    #         "labels": [[0, 1], [1, 0], [0, 1]]
    #     }
    #
    #     # Mock the behavior of the tokenizer.encode()
    #     mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    #
    #     # Call the preprocess_data function
    #     result = preprocess_data(example)
    #
    #     # Assert the expected output
    #     self.assertEqual(result, expected_output)
    #     pass

    @patch("huggingface.huggingface.prepare_training_datatests")
    @patch("huggingface.huggingface.AutoTokenizer.from_pretrained")
    @patch("transformers.Trainer")
    @patch("pandas.read_csv")
    @patch("pandas.DataFrame.to_csv")
    @mock.patch("os.listdir")
    def test_perform_training(self, mock_prepare_training_datatests, mock_tokenizer, mock_trainer, mock_read_csv, mock_to_csv,mock_listdir):
        # Mock necessary dependencies and set up any required return values

        mock_read_csv.return_value = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        mock_to_csv.return_value = None

        # Mock prepare_training_datatests
        mock_prepare_training_datatests.return_value = None

        # Mock AutoTokenizer.from_pretrained
        mock_tokenizer.return_value = ...  # Set up the return value of AutoTokenizer.from_pretrained

        # Mock Trainer
        mock_trainer.return_value = MagicMock()
        mock_trainer_instance = mock_trainer.return_value
        mock_trainer_instance.train.return_value = None
        mock_listdir.return_value = ['prp.csv']
        prepped_data_path = "../data/tmp/prepped_data"
        os.makedirs(prepped_data_path, exist_ok=True)

        status_output_path = "../data/tmp/status_output_data"
        os.makedirs(status_output_path, exist_ok=True)

        args = argparse.Namespace(prepped_data=prepped_data_path)
        sys.argv = ["program_name", "--prepped_data", prepped_data_path, "--status_output",status_output_path]

        # Call the function
        perform_training()

        # Assert that the necessary methods were called with the expected arguments
        mock_prepare_training_datatests.assert_called_once()
        mock_tokenizer.assert_called_once_with("yashveer11/final_model_category")
        mock_trainer.assert_called_once_with(
            model=...,  # Provide the expected model instance
            args=...,  # Provide the expected TrainingArguments instance
            train_dataset=...,  # Provide the expected train_dataset
            eval_dataset=...,  # Provide the expected eval_dataset
            tokenizer=...,  # Provide the expected tokenizer instance
            compute_metrics=...  # Provide the expected compute_metrics function
        )
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.push_to_hub.assert_called_once_with("End of training")

