import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    """Load the train and test CSV files."""
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Please download train.csv and test.csv into the 'data' folder.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """Clean missing data and encode categorical features safely."""
    # Combine datasets temporarily for uniform preprocessing
    target = train_df["Survived"]
    passenger_ids = test_df["PassengerId"]
    
    # Safely drop features using errors='ignore' so it won't crash if columns are missing
    train_df = train_df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"], errors='ignore')
    test_df = test_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors='ignore')
    
    # Handle missing values (Imputation)
    for df in [train_df, test_df]:
        if "Age" in df.columns:
            df["Age"] = df["Age"].fillna(train_df["Age"].median())
        if "Fare" in df.columns:
            df["Fare"] = df["Fare"].fillna(train_df["Fare"].median())
        if "Embarked" in df.columns and not train_df["Embarked"].mode().empty:
            df["Embarked"] = df["Embarked"].fillna(train_df["Embarked"].mode()[0])
    
    # Define our target training features
    features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked", "Age", "Fare"]
    
    # Keep only the features that actually exist in the current dataframes
    existing_features = [f for f in features if f in train_df.columns and f in test_df.columns]
    
    # Combine datasets dynamically using only existing columns
    combined = pd.concat([train_df[existing_features], test_df[existing_features]], axis=0)
    
    # Dynamically pick columns to encode (only if they exist in the dataset)
    encode_cols = [c for c in ["Sex", "Embarked"] if c in combined.columns]
    combined_encoded = pd.get_dummies(combined, columns=encode_cols, drop_first=True)
    
    # Split back into train and test
    X_train = combined_encoded.iloc[:len(train_df)]
    X_test = combined_encoded.iloc[len(train_df):]
    
    return X_train, target, X_test, passenger_ids

def main():
    print("🤖 Loading Titanic datasets...")
    train_df, test_df = load_data()
    
    print("🧼 Preprocessing and encoding features...")
    X, y, X_test, test_ids = preprocess_data(train_df, test_df)
    
    # local validation split to check model performance before submitting
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🏋️ Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate accuracy locally
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"✅ Local Validation Accuracy: {val_acc:.4f}")
    
    # Retrain on full training data and predict on Kaggle test set
    print("🔮 Generating predictions for Kaggle test dataset...")
    model.fit(X, y)
    final_predictions = model.predict(X_test)
    
    # Save the submission file
    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived": final_predictions
    })
    
    submission_path = os.path.join("data", "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"🎉 Success! Submission file saved to: {submission_path}")

if __name__ == "__main__":
    main()