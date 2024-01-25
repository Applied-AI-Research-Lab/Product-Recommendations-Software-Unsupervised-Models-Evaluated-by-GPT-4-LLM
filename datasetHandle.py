import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import os

class DatasetManager:
    def __init__(self):
        pass

    def download_and_rename_csv(self, url, local_file_path):
        # Make a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the content of the response to the local file
            with open(local_file_path, 'wb') as file:
                file.write(response.content)

            return [True, f"File downloaded and saved as {local_file_path}"]
        else:
            return [False, f"Failed to download the file. Status code: {response.status_code}"]

    def split_dataset(self, data_set, train_set, validation_set, test_set):
        df = pd.read_csv(data_set)  # Load the dataset

        # Split the dataset into training (70%), validation (15%), and test (15%) sets
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        # Save each split dataset to a CSV file
        train_df.to_csv(train_set, index=False)
        valid_df.to_csv(validation_set, index=False)
        test_df.to_csv(test_set, index=False)

        return [True, 'The dataset is split into training, validation, and test sets.']

    def check_csv_columns(self, data_set):
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(data_set)

            # Check if 'title' and 'categoryName' columns exist
            if 'title' in df.columns and 'categoryName' in df.columns:
                return [True, 'The columns title and categoryName exist']
            else:
                return [False, 'Missing columns title and categoryName']

        except pd.errors.EmptyDataError:
            return [False, f"The CSV file at {data_set} is empty."]
        except FileNotFoundError:
            return [False, f"The file at {data_set} does not exist."]
        except Exception as e:
            return [False, f"An error occurred: {e}"]

    def update_csv(self, file_path):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Update the values in the "kMeansEvaluation" column
        df['kMeansEvaluation'] = df['kMeansEvaluation'].astype(int)
        df['cbfEvaluation'] = df['cbfEvaluation'].astype(int)
        df['hierarchicalEvaluation'] = df['hierarchicalEvaluation'].astype(int)

        # Write the changes back to the CSV file
        df.to_csv(file_path, index=False)

    def evaluate_recommendations(self, file_path):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Sum the values of kMeansEvaluation, cbfEvaluation, hierarchicalEvaluation columns
        kmeans_sum = df['kMeansEvaluation'].sum()
        cbf_sum = df['cbfEvaluation'].sum()
        hierarchical_sum = df['hierarchicalEvaluation'].sum()

        # Determine which column has the greatest sum
        max_sum = max(kmeans_sum, cbf_sum, hierarchical_sum)

        # Return the result
        if max_sum == kmeans_sum:
            return "kmeans"
        elif max_sum == cbf_sum:
            return "cbf"
        else:
            return "hierarchical"

# # Create an instance of the class
# dataset_manager = DatasetManager()
#
# # Example
# dataset_manager.download_and_rename_csv('https://my-test-ecommerce.com/products.csv', 'Datasets/dataset.csv')
# dataset_manager.check_csv_columns('Datasets/dataset.csv')
# dataset_manager.split_dataset('Datasets/dataset.csv', 'Datasets/train_set.csv', 'Datasets/validation_set.csv', 'Datasets/test_set.csv')
