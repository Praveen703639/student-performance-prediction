from pathlib import Path

import pandas as pd
import joblib

from src.preprocessing.preprocess import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


MODEL_PATH = Path("models/random_forest_tuned.joblib")


def load_model():
    """Load the trained Random Forest pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_student(student_data: dict) -> float:
    """Predict the final grade for one student."""

    model = load_model()

    expected_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    missing_features = [
        feature
        for feature in expected_features
        if feature not in student_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    input_df = pd.DataFrame(
        [student_data],
        columns=expected_features,
    )

    prediction = model.predict(input_df)[0]

    # G3 is a grade from 0 to 20.
    prediction = max(0.0, min(20.0, float(prediction)))

    return prediction


if __name__ == "__main__":

    student = {
        "age": 17,
        "Medu": 3,
        "Fedu": 2,
        "traveltime": 1,
        "studytime": 3,
        "failures": 0,
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 3,
        "absences": 5,

        "school": "GP",
        "sex": "M",
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "A",
        "Mjob": "teacher",
        "Fjob": "services",
        "reason": "course",
        "guardian": "mother",
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
    }

    prediction = predict_student(student)

    print("\n" + "=" * 50)
    print("     STUDENT PERFORMANCE PREDICTOR")
    print("=" * 50)

    print(f"\nPredicted Final Grade (G3): {prediction:.2f} / 20")

    print("=" * 50)