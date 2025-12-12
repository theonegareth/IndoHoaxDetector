# Load necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths based on workspace structure
KOMPAS_RAW = "data/tweets_from_kompascom_20251106_124453.csv"
CEKFAKTA_RAW = "data/tweets_from_cekfaktacom_20251105_144313.csv"
KOMPAS_PRED = "prediction_results_tweets_from_kompascom_20251106_124453.csv"
CEKFAKTA_PRED = "prediction_results_tweets_from_cekfaktacom_20251105_144313.csv"  # Assuming similar naming

# Function to perform EDA on raw scraped data
def eda_raw_data(df, source_name):
    print(f"\n=== EDA for {source_name} Raw Data ===")
    print(f"Data shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nSample data:")
    print(df.head())
    
    # Basic stats
    print(f"\nNumber of unique tweets: {df['text'].nunique()}")
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    
    # Visualize tweet lengths
    df['text_length'] = df['text'].str.len()
    plt.figure(figsize=(10, 5))
    plt.hist(df['text_length'], bins=50, alpha=0.7)
    plt.xlabel('Tweet Length')
    plt.ylabel('Frequency')
    plt.title(f'Tweet Length Distribution - {source_name}')
    plt.show()
    
    # Top sources or users if available
    if 'username' in df.columns:
        top_users = df['username'].value_counts().head(10)
        print(f"\nTop 10 users by tweet count in {source_name}:")
        print(top_users)
        
        plt.figure(figsize=(10, 5))
        top_users.plot(kind='bar')
        plt.title(f'Top Users by Tweet Count - {source_name}')
        plt.show()

# Function to analyze prediction results
def analyze_predictions(df_pred, source_name):
    print(f"\n=== Prediction Analysis for {source_name} ===")
    print(f"Prediction data shape: {df_pred.shape}")
    print("Prediction columns:", df_pred.columns.tolist())
    print("\nSample predictions:")
    print(df_pred[['text', 'prediction', 'confidence_score']].head())
    
    # HOAX rate
    hoax_rate = (df_pred['prediction'] == 'HOAX').mean()
    print(f"\nOverall HOAX rate: {hoax_rate:.2%}")
    
    # Confidence distribution
    plt.figure(figsize=(10, 5))
    plt.hist(df_pred['confidence_score'], bins=20, alpha=0.7)
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title(f'Prediction Confidence Distribution - {source_name}')
    plt.show()
    
    # Confidence by prediction
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='prediction', y='confidence_score', data=df_pred)
    plt.title(f'Confidence by Prediction Type - {source_name}')
    plt.show()
    
    # Error analysis: Inspect low confidence predictions
    low_conf = df_pred[df_pred['confidence_score'] < 0.5]
    print(f"\nLow confidence predictions (<0.5): {len(low_conf)}")
    print("Sample low confidence:")
    print(low_conf[['text', 'prediction', 'confidence_score']].head(10))

# Load and analyze raw data
df_kompas_raw = pd.read_csv(KOMPAS_RAW)
eda_raw_data(df_kompas_raw, "Kompas")

df_cekfakta_raw = pd.read_csv(CEKFAKTA_RAW)
eda_raw_data(df_cekfakta_raw, "CekFakta")

# Load and analyze predictions
df_kompas_pred = pd.read_csv(KOMPAS_PRED)
analyze_predictions(df_kompas_pred, "Kompas")

# Assuming CekFakta predictions exist; if not, handle gracefully
try:
    df_cekfakta_pred = pd.read_csv(CEKFAKTA_PRED)
    analyze_predictions(df_cekfakta_pred, "CekFakta")
except FileNotFoundError:
    print("\nCekFakta prediction results not found. Skipping analysis.")

# Comparative analysis
print("\n=== Comparative HOAX Rates ===")
kompas_hoax = (df_kompas_pred['prediction'] == 'HOAX').mean()
try:
    cekfakta_hoax = (df_cekfakta_pred['prediction'] == 'HOAX').mean()
    print(f"Kompas HOAX rate: {kompas_hoax:.2%}")
    print(f"CekFakta HOAX rate: {cekfakta_hoax:.2%}")
except NameError:
    print(f"Kompas HOAX rate: {kompas_hoax:.2%} (CekFakta not available)")