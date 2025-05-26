import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load cleaned dataset
df = pd.read_csv("vidhuk_reviews_cleaned_extra_5.csv", encoding="utf-8-sig")

# Detect relevant columns
text_column = next((col for col in df.columns if "cleaned_text" in col.lower()), None)
stars_column = next((col for col in df.columns if "stars" in col.lower()), None)

# Count number of cleaned reviews per star rating
review_counts = Counter(df[stars_column].dropna())

# -------- First Graph: Full Distribution --------
ratings = [1, 2, 3, 4, 5]
counts = [review_counts.get(r, 0) for r in ratings]

plt.figure(figsize=(8, 5))
bars = plt.bar(ratings, counts, color='skyblue', edgecolor='black')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, str(height),
             ha='center', va='bottom', fontsize=10)

plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.title("Distribution of Star Ratings")
plt.xticks(ratings)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("cleaned rews distribution.png")
plt.show()

# -------- Second Graph: 1-star vs (4+5)-star --------
count_1 = review_counts.get(1, 0)
count_4_5 = review_counts.get(4, 0) + review_counts.get(5, 0)

plt.figure(figsize=(6, 4))
bars2 = plt.bar(['1 star', '4+5 stars'], [count_1, count_4_5],
                color=['crimson', 'seagreen'], edgecolor='black')

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, str(height),
             ha='center', va='bottom', fontsize=10)

plt.xlabel("Review Type")
plt.ylabel("Number of Reviews")
plt.title("Negative vs Positive Cleaned Reviews")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("ratings_1_vs_4_5.png")
plt.show()
