import configparser

config = configparser.ConfigParser()
config.read('config.ini')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

####################################################################################################
# This script reads 'summarization.csv' generated from 'test_grammar_model.py'.
# It plots graphs showing AI model output vs original sentence and target sentence
####################################################################################################


# Load the data
df = pd.read_csv('History/1/summarization.csv')

original_result_average = df[['Grammar_OrigRes', 'Semantic_OrigRes', 'Keywords_OrigRes']].mean()
target_result_average = df[['Grammar_TargRes', 'Semantic_TargRes', 'Keywords_TargRes']].mean()

# Create a DataFrame for plotting
metrics = ['Grammar', 'Semantic', 'Keywords']
mean_scores = pd.DataFrame({
    'Metric': metrics * 2,
    'Score': list(original_result_average) + list(target_result_average),
    'Comparison': ['Original vs Result'] * 3 + ['Target vs Result'] * 3
})

# Plot 1: Bar chart of average scores
plt.figure(figsize=(10, 6))
sns.barplot(x='Metric', y='Score', hue='Comparison', data=mean_scores, palette='viridis')
plt.title('Average Performance Metrics')
plt.ylim(0, 1.1)
plt.savefig('performance_metrics.png', bbox_inches='tight')
plt.close()

# Plot 2: Scatter plot of Semantic vs Keyword scores
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['Semantic_OrigRes'], y=df['Keywords_OrigRes'],
    label='Original vs Result', color='blue', alpha=0.7
)
sns.scatterplot(
    x=df['Semantic_TargRes'], y=df['Keywords_TargRes'],
    label='Target vs Result', color='orange', alpha=0.7
)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # Diagonal reference line
plt.xlabel('Semantic Similarity')
plt.ylabel('Keyword Overlap')
plt.title('Semantic vs Keyword Scores')
plt.legend()
plt.savefig('semantic_vs_keywords.png', bbox_inches='tight')
plt.close()

print("Graphs saved as performance_metrics.png and semantic_vs_keywords.png")