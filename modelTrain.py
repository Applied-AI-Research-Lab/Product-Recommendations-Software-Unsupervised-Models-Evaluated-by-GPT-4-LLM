import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import linear_kernel
import csv
import os
import numpy as np


class ModelTrain:
    def __init__(self):
        self.title_tfidf = None
        self.category_tfidf = None

    def transform_input(self, title, category):
        title_transformed = self.title_tfidf.transform([title])
        category_transformed = self.category_tfidf.transform([category])
        return hstack([title_transformed, category_transformed])

    def train_tfidf_transformers(self, train_data):
        self.title_tfidf = TfidfVectorizer().fit(train_data['title'])
        self.category_tfidf = TfidfVectorizer().fit(train_data['categoryName'])

    def recommend_product_cbf(self, title, category, train_data, num_recommendations=1):
        input_features = self.transform_input(title, category)
        train_data_transformed = hstack([
            self.title_tfidf.transform(train_data['title']),
            self.category_tfidf.transform(train_data['categoryName'])
        ])

        similarity_scores = linear_kernel(input_features.toarray(), train_data_transformed.toarray()).flatten()
        sorted_products = np.argsort(similarity_scores)[::-1]
        recommended_titles = [train_data.iloc[index]['title'] for index in sorted_products]
        return recommended_titles[:num_recommendations]

    def get_cbf_recommendations(self, train_data, recommendation_column):
        file_path = "Datasets/recommendations.csv"
        temp_file_path = "Datasets/temp_recommendations.csv"

        self.train_tfidf_transformers(train_data)

        with open(file_path, 'r') as infile, open(temp_file_path, 'w', newline='') as outfile:
            csv_reader = csv.DictReader(infile)
            fieldnames = csv_reader.fieldnames

            if recommendation_column not in fieldnames:
                fieldnames.append(recommendation_column)

            csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            csv_writer.writeheader()

            for row in csv_reader:
                title = row['title']
                category = row['categoryName']
                recommended_titles = self.recommend_product_cbf(title, category, train_data, 1)
                prediction_value = recommended_titles[0]

                row[recommendation_column] = prediction_value
                csv_writer.writerow(row)

        os.replace(temp_file_path, file_path)

    def train_hierarchical_model(self, train_data, val_data):
        combined_data = pd.concat([train_data, val_data], axis=0, ignore_index=True)
        X_combined = combined_data[['title', 'categoryName']]

        self.title_tfidf = TfidfVectorizer().fit(X_combined['title'])
        self.category_tfidf = TfidfVectorizer().fit(X_combined['categoryName'])

        title_transformed = self.title_tfidf.transform(X_combined['title'])
        category_transformed = self.category_tfidf.transform(X_combined['categoryName'])
        X_combined_transformed = hstack([title_transformed, category_transformed])

        num_clusters = 2
        model = AgglomerativeClustering(n_clusters=num_clusters)

        model.fit(X_combined_transformed.toarray())

        joblib.dump((model, self.title_tfidf, self.category_tfidf), 'Datasets/hierarchical_model.joblib')

        predictions = model.labels_
        silhouette_avg = silhouette_score(X_combined_transformed.toarray(), predictions)
        print(f"Hierarchical Silhouette Score: {silhouette_avg}")

    def hierarchical_recommend_product(self, title, category, model, train_data_combined, X_combined_transformed,
                                       num_recommendations=1):
        input_features = self.transform_input(title, category)
        cluster = model.labels_

        cluster_indices = np.where(cluster == cluster[0])[0]

        similarity_scores = []
        for index in cluster_indices:
            similarity_score = np.dot(input_features.toarray(), X_combined_transformed[index].toarray().T)[0, 0]
            similarity_scores.append((index, similarity_score))

        sorted_products = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        recommended_titles = [train_data_combined.iloc[index]['title'] for index, _ in sorted_products]

        return recommended_titles[:num_recommendations]

    def get_hierarchical_recommendations(self, model_path, train_data_combined, recommendation_column,
                                         X_combined_transformed):
        model, title_tfidf, category_tfidf = joblib.load(model_path)

        recommended_titles = self.hierarchical_recommend_product(
            train_data_combined['title'][0],
            train_data_combined['categoryName'][0],
            model,
            train_data_combined,
            X_combined_transformed,
            1
        )

        print(f'Hierarchical Recommended titles: {recommended_titles}')

        file_path = "Datasets/recommendations_hierarchical.csv"
        temp_file_path = "Datasets/temp_recommendations_hierarchical.csv"

        with open(file_path, 'r') as infile, open(temp_file_path, 'w', newline='') as outfile:
            csv_reader = csv.DictReader(infile)
            fieldnames = csv_reader.fieldnames

            if recommendation_column not in fieldnames:
                fieldnames.append(recommendation_column)

            csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            csv_writer.writeheader()

            for row in csv_reader:
                title = row['title']
                category = row['categoryName']
                recommended_titles = self.hierarchical_recommend_product(title, category, model, train_data_combined,
                                                                         X_combined_transformed, 1)
                prediction_value = recommended_titles[0]

                row[recommendation_column] = prediction_value
                csv_writer.writerow(row)

        os.replace(temp_file_path, file_path)

    def kmeans(self, train_data, val_data, num_clusters=10):
        combined_data = pd.concat([train_data, val_data], axis=0, ignore_index=True)
        X_combined = combined_data[['title', 'categoryName']]

        self.title_tfidf = TfidfVectorizer().fit(X_combined['title'])
        self.category_tfidf = TfidfVectorizer().fit(X_combined['categoryName'])

        title_transformed = self.title_tfidf.transform(X_combined['title'])
        category_transformed = self.category_tfidf.transform(X_combined['categoryName'])
        X_combined_transformed = hstack([title_transformed, category_transformed])

        model = make_pipeline(KMeans(n_clusters=num_clusters))
        model.fit(X_combined_transformed)

        joblib.dump((model, self.title_tfidf, self.category_tfidf), 'Datasets/k_means_model.joblib')

        X_val_transformed = hstack(
            [self.title_tfidf.transform(val_data['title']), self.category_tfidf.transform(val_data['categoryName'])])
        predictions = model.predict(X_val_transformed)

        silhouette_avg = silhouette_score(X_combined_transformed, model.named_steps['kmeans'].labels_)
        print(f"KMeans Silhouette Score: {silhouette_avg}")

    def kmeans_recommendations(self, model_path, train_data, recommendation_column, X_combined_transformed,
                               num_recommendations=1):
        model, title_tfidf, category_tfidf = joblib.load(model_path)

        input_features = self.transform_input(train_data['title'][0], train_data['categoryName'][0])
        cluster = model.named_steps['kmeans'].predict(input_features)

        labels = model.named_steps['kmeans'].labels_
        cluster_indices = np.where(labels == cluster[0])[0]

        similarity_scores = []
        for index in cluster_indices:
            similarity_score = np.dot(input_features.toarray(), X_combined_transformed[index].toarray().T)[0, 0]
            similarity_scores.append((index, similarity_score))

        sorted_products = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        recommended_titles = [train_data.iloc[index]['title'] for index, _ in sorted_products]

        return recommended_titles[:num_recommendations]

# # Example:
# # Load the training and validation datasets
# train_data = pd.read_csv('Datasets/train_set.csv')
# validation_data = pd.read_csv('Datasets/validation_set.csv')
#
# # Initialize the ModelTrain class
# model_trainer = ModelTrain()
#
# # Train the CBF model
# model_trainer.train_tfidf_transformers(train_data)
#
# # Get CBF recommendations
# model_trainer.get_cbf_recommendations(train_data, 'cbfRecommendation')
#
# # Train the KMeans model
# model_trainer.kmeans(train_data, validation_data)
#
# # Get KMeans recommendations
# model_trainer.kmeans_recommendations('Datasets/k_means_model.joblib', train_data, 'kmeansRecommendation', X_combined_transformed)
#
# # Train the Hierarchical model
# model_trainer.train_hierarchical_model(train_data, validation_data)
#
# # Get Hierarchical recommendations
# model_trainer.get_hierarchical_recommendations('Datasets/hierarchical_model.joblib', train_data_combined, 'hierarchicalRecommendation', X_combined_transformed)
