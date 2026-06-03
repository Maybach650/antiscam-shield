import json
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ===== PATHS =====
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "antiscam_dataset_300.json"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ===== LOAD DATA =====
print("Loading dataset...")
with open(DATA_FILE, encoding="utf-8") as f:
    data = json.load(f)

texts = [d["dialogue"] for d in data]
labels = [d["label"] for d in data]

print(f"Total samples: {len(texts)}")
print(f"Scam: {sum(labels)} | Normal: {len(labels) - sum(labels)}")

# ===== SPLIT DATA =====
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ===== VECTORIZE =====
print("\nVectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    analyzer="char_wb",
    min_df=1
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===== TRAIN MODEL =====
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# ===== EVALUATE =====
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Scam"]))

# ===== SAVE MODEL =====
model_path = MODEL_DIR / "scam_detector.pkl"
vectorizer_path = MODEL_DIR / "vectorizer.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

print(f"Model saved to: {model_path}")
print(f"Vectorizer saved to: {vectorizer_path}")

# ===== TEST WITH EXAMPLES =====
print("\n" + "="*50)
print("TESTING WITH EXAMPLES")
print("="*50)

examples = [
    "Scammer: Salam, men bankdan jaň edýärin. PIN kodуňyzy aýdyň.\nVictim: PIN kodum 1234.",
    "Caller: Salam dost, bu gün futbola gidýäňmi?\nDost: Hawa, sagat 7-de.",
    "Scammer: Здравствуйте, с вашей карты снимают деньги! Назовите CVV.\nVictim: CVV 123.",
    "Caller: Привет, ты уже поел?\nPerson: Да, только поужинал.",
]

for example in examples:
    vec = vectorizer.transform([example])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "🚨 SCAM" if pred == 1 else "✅ NORMAL"
    confidence = prob[pred] * 100
    print(f"\n{label} ({confidence:.1f}% confidence)")
    print(f"Text: {example[:60]}...")
