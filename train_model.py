import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle, os

# Sample phishing dataset (replace with a better one later)
# df = pd.read_csv("https://raw.githubusercontent.com/s4dush/phishing-dataset/main/phishing.csv")
df = pd.read_csv("data/phishing_dataset.csv")

# Optional: Save locally (in case you want to view it in VS Code)
os.makedirs("data", exist_ok=True)
df.to_csv("data/phishing_dataset.csv", index=False)

# Features and target
X = df['Email Text']
y = df['Label']  # 1 = phishing, 0 = legitimate

# Vectorize the email content
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
with open("model/phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
    
print("Model trained and saved successfully.")