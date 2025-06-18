import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("vidhuk_reviews_with_sentiment.csv")


df['match/nomatch'] = df['match/nomatch'].fillna('missing')


df['rate by program'] = pd.to_numeric(df['rate by program'], errors='coerce')  
df['rate by program'] = df['rate by program'].fillna(0)

def analyze_reviews(df):
    total_reviews = len(df)
    avg_star_rating = df['stars'].mean()
    avg_program_rating = df['rate by program'].mean()  

    df['cleaned_len'] = df['cleaned_text'].astype(str).apply(len)
    text_len_mean = df['cleaned_len'].mean()
    text_len_min = df['cleaned_len'].min()
    text_len_max = df['cleaned_len'].max()

    star_dist = df['stars'].value_counts().sort_index()
    match_dist = df['match/nomatch'].value_counts()

    df['review_len'] = df['review'].astype(str).apply(len)
    df['reduction_pct'] = (1 - df['cleaned_len'] / df['review_len']) * 100
    reduction_mean = df['reduction_pct'].mean()

    missing_values = df.isnull().sum()
    duplicate_count = df.duplicated().sum()


    print(f"Total reviews count: {total_reviews}")
    print(f"Average star rating: {avg_star_rating:.3f}")
    print(f"Average program rating: {avg_program_rating:.3f}")
    print(f"\nText length (cleaned_text) - mean: {text_len_mean:.1f}, min: {text_len_min}, max: {text_len_max}")
    print("\nStar rating distribution:")
    print(star_dist.to_string())
    print("\nMatch/No-match distribution:")
    print(match_dist.to_string())
    print(f"\nAverage text length reduction percentage: {reduction_mean:.2f}%")
    print("\nData quality metrics:")
    print(f"Missing values per column:\n{missing_values}")
    print(f"Number of duplicate rows: {duplicate_count}")
    print(df[df.duplicated(keep=False)] 
)


analyze_reviews(df)
