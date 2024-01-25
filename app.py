from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import json
import pandas as pd

from datasetHandle import DatasetManager
from modelTrain import ModelTrain
from gptEvaluation import GptEvaluation

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/', methods=['POST', 'GET'])
def all_my_projects():
    return 'Project Links'


@app.route('/recommendations-train', methods=['POST', 'GET'])
def recommendations_train():
    csv = ''
    if request.method == 'GET':
        # Handle the GET request and access the query parameters.
        csv = request.args.get('csv', '')
    elif request.method == 'POST':
        # Handle the POST request and access the query parameters.
        data = request.get_json()
        csv = data.get('csv', '')

    if not csv:
        # If post data is missing, return an error
        return jsonify({'status': False, 'data': 'Missing required query parameters'})

    # Instantiate the DatasetManager class
    dataset_manager = DatasetManager()

    # Download the CSV
    download = dataset_manager.download_and_rename_csv(csv, 'Datasets/dataset.csv')
    if not download[0]:
        return download

    # Check the CSV columns.
    columns = dataset_manager.check_csv_columns('Datasets/dataset.csv')
    if not columns[0]:
        return columns

    # Split the provided CSV into training, validation, and test sets.
    split = dataset_manager.split_dataset('Datasets/dataset.csv', 'Datasets/train_set.csv',
                                          'Datasets/validation_set.csv', 'Datasets/test_set.csv')
    if not split[0]:
        return split

    # Train the unsupervised models used for generating product recommendations.
    model_trainer = ModelTrain()
    train_data = pd.read_csv('Datasets/train_set.csv')
    validation_data = pd.read_csv('Datasets/validation_set.csv')

    # Train the KMeans model
    model_trainer.kmeans(train_data, validation_data)

    # Train the CBF model
    model_trainer.train_tfidf_transformers(train_data)

    # Train the Hierarchical model
    model_trainer.train_hierarchical_model(train_data, validation_data)

    # Run the trained models to generate product recommendations.
    # Get KMeans recommendations
    model_trainer.kmeans_recommendations('Datasets/k_means_model.joblib', train_data, 'kmeansRecommendation')

    # Get CBF recommendations
    model_trainer.get_cbf_recommendations(train_data, 'cbfRecommendation')

    # Get Hierarchical recommendations
    model_trainer.get_hierarchical_recommendations('Datasets/hierarchical_model.joblib', train_data,
                                                   'hierarchicalRecommendation')

    # Evaluate models' recommendations using the GPT model
    model_id = 'gpt-4'
    gpt_evaluator = GptEvaluation(model_id)
    gpt_evaluator.evaluate('kMeans', 'Datasets/recommendations.csv')
    gpt_evaluator.evaluate('cbf', 'Datasets/recommendations.csv')
    gpt_evaluator.evaluate('hierarchical', 'Datasets/recommendations.csv')

    # Standardize the values in the evaluation columns by transforming the values 1.0 and 0.0 to 1 and 0, respectively.
    dataset_manager.update_csv('Datasets/recommendations.csv')

    # Sum the values in each evaluation column, and return the model that is most suitable for product recommendations.
    return dataset_manager.evaluate_recommendations('Datasets/recommendations.csv')


@app.route('/recommendations-run', methods=['POST', 'GET'])
def recommendations_run():
    csv = ''
    if request.method == 'GET':
        # Handle the GET request and access the query parameters.
        csv = request.args.get('model', '')
        csv = request.args.get('title', '')
        csv = request.args.get('category', '')
    elif request.method == 'POST':
        # Handle the POST request and access the query parameters.
        data = request.get_json()
        csv = data.get('model', '')
        csv = data.get('title', '')
        csv = data.get('category', '')

    # Use the model, title, and category to make recommendations using the ModelTrain class.
    # Return recommendations in JSON format.
