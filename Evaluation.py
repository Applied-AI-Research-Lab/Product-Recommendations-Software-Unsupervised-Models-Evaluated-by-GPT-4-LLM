import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_evaluations(df, kmeans_col='kMeansEvaluation', cbf_col='cbfEvaluation', hierarchical_col='hierarchicalEvaluation'):
    # Count the success and failure for each model
    kmeans_counts = df[kmeans_col].value_counts()
    cbf_counts = df[cbf_col].value_counts()
    hierarchical_counts = df[hierarchical_col].value_counts()

    # Bar chart with seaborn
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.2
    indices = [0, 1]

    ax.bar(indices, kmeans_counts, width, label='kMeans', color='skyblue', align='center', alpha=0.7)
    ax.bar([i + width for i in indices], cbf_counts, width, label='cbf', color='salmon', align='center', alpha=0.7)
    ax.bar([i + 2 * width for i in indices], hierarchical_counts, width, label='Hierarchical', color='lightgreen', align='center', alpha=0.7)

    # Add percentages on top of the bars
    total_samples = len(df)
    for i, counts in enumerate([kmeans_counts, cbf_counts, hierarchical_counts]):
        for j, count in enumerate(counts):
            percentage = count / total_samples * 100
            ax.text(i * width + j, count + 0.2, f'{percentage:.2f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks([i + width for i in indices])
    ax.set_xticklabels(['Succeed', 'Failed'])

    ax.set_xlabel('GPT-4 Evaluation')
    ax.set_ylabel('Count')
    ax.set_title('Comparison of Product Recommender Models')

    ax.legend()

    plt.show()

df = pd.read_csv('Datasets/recommendations.csv')
compare_evaluations(df)
