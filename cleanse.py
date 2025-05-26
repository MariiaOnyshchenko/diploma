import pandas as pd
import spacy
import re
from collections import Counter
import matplotlib.pyplot as plt

# Load spaCy model for Ukrainian (or Russian if needed)
nlp = spacy.load("uk_core_news_sm")  # or "ru_core_news_sm"

# Load dataset
df = pd.read_csv("vidhuk_reviews copy extra 5.csv", encoding="utf-8")

# Detect relevant column names
text_column = next((col for col in df.columns if "review" in col.lower()), None)
stars_column = next((col for col in df.columns if "stars" in col.lower()), None)

if not text_column or not stars_column:
    raise ValueError("Couldn't identify review or stars columns. Please check column names.")

# Function to clean review text using NER
def clean_text(text):
    if pd.isnull(text):
        return text
    doc = nlp(text)
    
    # Mask named entities
    for ent in doc.ents:
        text = text.replace(ent.text, "<ENTITY>")
    
    # Remove long numbers and alphanumeric codes
    text = re.sub(r"\b\d{4,}\b", "<NUM>", text)  # 4+ digit numbers
    text = re.sub(r"\b[A-Z0-9]{8,}\b", "<ORDER_ID>", text)  # long alphanum blocks
    text = re.sub(r"\b\d+[A-Z]+\d*[A-Z]*\b", "<ORDER_ID>", text)  # like MGRAE0013273319YQ
    text = re.sub(r"\b[A-Z]{3,}\d{3,}[A-Z]*\b", "<ORDER_ID>", text)  # like NEYTMJM250414558
    
    return text


# Clean reviews
df['cleaned_text'] = df[text_column].apply(clean_text)

# Count number of reviews by star rating
rating_counts = Counter(df[stars_column].dropna())

# Save cleaned data
df.to_csv("vidhuk_reviews_cleaned_extra_5.csv", index=False, encoding="utf-8-sig")

# Plot the statistics
ratings = [1, 2, 3, 4, 5]
counts = [rating_counts.get(r, 0) for r in ratings]

plt.figure(figsize=(8, 5))
bars = plt.bar(ratings, counts, color='skyblue', edgecolor='black')

# ✅ Add numbers on top of bars
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
plt.savefig("ratings_distribution.png")
plt.show()