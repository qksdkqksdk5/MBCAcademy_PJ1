# analysis/accident/model.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

FEATURES = [
    '발생건수(건)', '부상자수(명)',
    '사고당사망률', '사고당부상률', 'CCTV설치대수'
]

def train_model(df):
    X = df[FEATURES]
    y = df['심각정도']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])

    pipe.fit(X_train, y_train)

    return pipe, le, X_test, y_test


def evaluate_model(pipe, X_test, y_test, le):
    y_pred = pipe.predict(X_test)

    return {
        "accuracy": pipe.score(X_test, y_test),
        "report": classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            zero_division=0
        ),
        "confusion": confusion_matrix(y_test, y_pred)
    }


def predict_severity(pipe, le, sample_dict):
    df = pd.DataFrame([sample_dict])
    code = pipe.predict(df)[0]
    return le.inverse_transform([code])[0]
