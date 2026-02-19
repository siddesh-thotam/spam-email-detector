import pandas as pd
import string
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nltk.download('stopwords')

# 1. Load Dataset
df = pd.read_csv("spam.csv", encoding="latin1")
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
df = df.rename(columns={"v1": "label", "v2": "message"})
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 2. Text Cleaning Function
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    text = ''.join(char for char in text if not char.isdigit())
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df["cleaned_message"] = df["message"].apply(clean_text)

# 3. Vectorization (TF-IDF + N-grams)
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 3),
    max_features=5000
)

X = vectorizer.fit_transform(df["cleaned_message"])
y = df["label"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Naive Bayes
from sklearn.naive_bayes import ComplementNB

model = ComplementNB()
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)

print("\nModel Evaluation:\n")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# 7. Save Model & Vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")

# 8. Manual Testing Mode
def predict_spam(message):
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0][1]

    if prediction == 1:
        return f"Spam 🚨 (Confidence: {probability:.2f})"
    else:
        return f"Ham ✅ (Confidence: {1 - probability:.2f})"



print("\nEnter a message to test (type 'exit' to quit):")

while True:
    user_input = input(">> ")
    if user_input.lower() == "exit":
        break
    print(predict_spam(user_input))