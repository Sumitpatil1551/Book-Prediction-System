import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load the data (you would need to create or obtain this dataset)
books = pd.read_csv("book_dataset.csv")

# Display the first few rows and basic information about the dataset
print(books.head())
print(books.info())

# Check for null values and duplicates
print(books.isnull().sum())
print(f"Number of duplicate rows: {books.duplicated().sum()}")

# Display statistical summary
print(books.describe())

# Create a correlation heatmap (excluding non-numeric columns)
numeric_columns = ['page_count', 'avg_rating', 'num_reviews', 'publication_year']
correlation = books[numeric_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cbar=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Display value counts of the 'genre' column
print(books['genre'].value_counts())
print(f"Number of unique genres: {books['genre'].nunique()}")

# Create distribution plots for numerical features
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(books[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

# Create a dictionary to map genre names to numbers
genre_dict = {genre: i+1 for i, genre in enumerate(books['genre'].unique())}
books['genre_encoded'] = books['genre'].map(genre_dict)

# Separate features and target
X = books.drop(['genre', 'genre_encoded', 'title', 'author'], axis=1)
y = books['genre_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
mx = MinMaxScaler()
X_train_scaled = mx.fit_transform(X_train)
X_test_scaled = mx.transform(X_test)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train_scaled)
X_test_scaled = sc.transform(X_test_scaled)

# Train and evaluate RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
print(f"RandomForestClassifier accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Define recommendation function
def recommend_book(page_count, avg_rating, num_reviews, publication_year):
    features = np.array([[page_count, avg_rating, num_reviews, publication_year]])
    mx_features = mx.transform(features)
    sc_mx_features = sc.transform(mx_features)
    prediction = rf_model.predict(sc_mx_features)[0]
    recommended_genre = list(genre_dict.keys())[list(genre_dict.values()).index(prediction)]
    return recommended_genre

# Example prediction
page_count, avg_rating, num_reviews, publication_year = 300, 4.2, 1000, 2020
predicted_genre = recommend_book(page_count, avg_rating, num_reviews, publication_year)
print(f"Recommended genre: {predicted_genre}")

# Save models and scalers
pickle.dump(rf_model, open('book_model.pkl', 'wb'))
pickle.dump(mx, open('book_minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('book_standscaler.pkl', 'wb'))
pickle.dump(genre_dict, open('genre_dict.pkl', 'wb'))

print("Models and scalers saved successfully!")
