import argparse
import seaborn as sns
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("dark_background")

nltk.download("stopwords")
stop_words = set(stopwords.words("indonesian"))


def count_word_frequencies(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    words = [word for word in words if word not in stop_words]
    word_counts = Counter(words)
    return word_counts


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="input text file path")
ap.add_argument("-o", "--output", required=True,
                help="output word frequency txt path")
ap.add_argument("-d", "--data", required=True,
                help="data capres/cawapres")
args = vars(ap.parse_args())

input_file_path = args["input"]
with open(input_file_path, "r", encoding="utf-8") as file:
    transcribed_text = file.read()

word_counts = count_word_frequencies(transcribed_text)

sorted_word_counts = sorted(
    word_counts.items(), key=lambda x: x[1], reverse=True)

output_file_path = args["output"]
with open(output_file_path, "w", encoding="utf-8") as file:
    for word, count in sorted_word_counts:
        file.write(f"{word}: {count}\n")

print(
    f"Word frequencies have been saved to '{output_file_path}' (sorted by count in descending order)")

# Generate and display word cloud
wordcloud = WordCloud(width=800, height=400,
                      stopwords=stop_words,
                      colormap='Set3',
                      background_color="black",
                      ).generate_from_frequencies(word_counts)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig(f"05_data_word_cloud/{args['data']}.png", bbox_inches="tight")
plt.show()

word_freq_df = pd.DataFrame(word_counts.items(), columns=[
                            'Word', 'Frequency']).sort_values(by="Frequency", ascending=False)

# Plotting the barplot
plt.figure(figsize=(10, 8))
barplot = sns.barplot(x='Frequency', y='Word',
                      data=word_freq_df.head(20), palette='Blues_d')
plt.title(f'Top 20 Word Frequencies from {args["data"].capitalize()}')
plt.xlabel('Frequency')
plt.ylabel('Word')
for index, (word, frequency) in enumerate(word_freq_df.head(20).values):
    plt.text(frequency, index, f'{frequency}', va='center', fontsize=10, bbox=dict(
        facecolor='white', alpha=0.5))
sns.despine(left=True, bottom=True)
plt.savefig(
    f"06_data_word_frequency/{args['data']}.png", bbox_inches="tight")
plt.show()
