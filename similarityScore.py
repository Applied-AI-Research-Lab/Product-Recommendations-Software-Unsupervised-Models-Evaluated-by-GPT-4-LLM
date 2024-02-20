import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('Datasets/recommendations.csv')

def calculate_similarity(row, product, recommendation_product):  # calculate cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([row[product], row[recommendation_product]])
    similarity_matrix = cosine_similarity(vectors)
    similarity_score = similarity_matrix[0, 1]
    return round(similarity_score, 4)

# Apply the results into SS columns
df['kMeansEvaluationSS'] = df.apply(lambda row: calculate_similarity(row, 'title', 'kMeansRecommendation'), axis=1)
df['cbfEvaluationSS'] = df.apply(lambda row: calculate_similarity(row, 'title', 'cbfRecommendation'), axis=1)
df['hierarchicalEvaluationSS'] = df.apply(lambda row: calculate_similarity(row, 'title', 'hierarchicalRecommendation'), axis=1)

# df.to_csv('Datasets/recommendations_with_similarity_scores.csv', index=False) # Save the updated DataFrame to a new CSV file

# Descriptive statistics
print(df['kMeansEvaluationSS'].describe())
print(df['cbfEvaluationSS'].describe())
print(df['hierarchicalEvaluationSS'].describe())




