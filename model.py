"""
AI Model Training Module (Upgraded)
Trains RandomForestRegressor for water quality risk prediction.
Uses the advanced normalized deviation risk index.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

from simulation import generate_training_dataset, calculate_risk_index


# Model configuration - includes conductivity and temporal features
MODEL_FEATURES = ['temperature', 'flow', 'pressure', 'chlorine', 'pH', 'turbidity', 'conductivity', 'hour', 'is_weekend']
MODEL_PATH = 'water_risk_model.joblib'


def prepare_training_data(n_samples: int = 5000) -> tuple:
    """
    Generate and prepare training data with calculated risk indices.
    
    Args:
        n_samples: Number of training samples
    
    Returns:
        Tuple of (X, y, df)
    """
    print(f"Generating {n_samples} training samples...")
    df = generate_training_dataset(n_samples=n_samples, include_anomalies=True)
    
    # Calculate risk index for each row using the advanced formula
    print("Calculating risk indices with normalized deviation model...")
    risk_indices = []
    for _, row in df.iterrows():
        risk, _ = calculate_risk_index(row.to_dict())
        risk_indices.append(risk)
    
    df['risk_index'] = risk_indices
    
    X = df[MODEL_FEATURES]
    y = df['risk_index']
    
    return X, y, df


def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
    """
    Train RandomForestRegressor model.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Fraction of data for testing
    
    Returns:
        Tuple of (trained_model, test_metrics, feature_importance)
    """
    print(f"Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    print("\nTraining RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2_score': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    feature_importance = dict(zip(MODEL_FEATURES, model.feature_importances_))
    
    return model, metrics, feature_importance


def save_model(model, path: str = MODEL_PATH):
    """Save trained model to disk."""
    print(f"\nSaving model to {path}...")
    joblib.dump(model, path)
    print("Model saved successfully!")


def load_model(path: str = MODEL_PATH):
    """Load trained model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Please train the model first.")
    return joblib.load(path)


def predict_risk(input_data: dict, model=None) -> float:
    """
    Predict risk index for a single input row.
    
    Args:
        input_data: Dictionary with sensor readings
        model: Trained model (loads from disk if None)
    
    Returns:
        Predicted risk index (0-100)
    """
    if model is None:
        model = load_model()
    
    features = pd.DataFrame([{k: input_data.get(k, 0) for k in MODEL_FEATURES}])
    prediction = model.predict(features)[0]
    prediction = np.clip(prediction, 0, 100)
    
    return round(prediction, 1)


def main():
    """Main function to train and save the model."""
    print("=" * 60)
    print("Water Quality Risk Prediction Model Training (Upgraded)")
    print("Using Normalized Deviation + Interaction Risk Model")
    print("=" * 60 + "\n")
    
    X, y, df = prepare_training_data(n_samples=10000)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
    print(f"  Risk Index range: {y.min():.1f} - {y.max():.1f}")
    print(f"  Risk Index mean: {y.mean():.1f}")
    print(f"  Risk Index std: {y.std():.1f}")
    
    model, metrics, importance = train_model(X, y)
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"  R² Score:           {metrics['r2_score']:.4f}")
    print(f"  Mean Absolute Error: {metrics['mae']:.2f}")
    print(f"  Root Mean Sq Error:  {metrics['rmse']:.2f}")
    
    print("\nFeature Importance:")
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(imp * 40)
        print(f"  {feature:12} : {imp:.4f} {bar}")
    
    save_model(model)
    
    print("\n" + "=" * 60)
    print("TEST PREDICTIONS")
    print("=" * 60)
    
    # Normal reading (daytime, weekday)
    normal_data = {
        'temperature': 20.0, 'flow': 2.5, 'pressure': 4.0,
        'chlorine': 1.2, 'pH': 7.2, 'turbidity': 0.4,
        'conductivity': 380, 'hour': 14, 'is_weekend': 0
    }
    pred = predict_risk(normal_data, model)
    print(f"\nNormal reading: Risk = {pred}")
    
    # Moderate risk (night, higher temporal risk)
    moderate_data = {
        'temperature': 24.0, 'flow': 1.5, 'pressure': 3.2,
        'chlorine': 0.6, 'pH': 7.0, 'turbidity': 0.8,
        'conductivity': 480, 'hour': 3, 'is_weekend': 0
    }
    pred = predict_risk(moderate_data, model)
    print(f"Moderate risk: Risk = {pred}")
    
    # High risk (triggers interaction risks, high conductivity)
    high_risk_data = {
        'temperature': 28.0, 'flow': 0.8, 'pressure': 2.3,
        'chlorine': 0.3, 'pH': 6.3, 'turbidity': 3.5,
        'conductivity': 650, 'hour': 2, 'is_weekend': 1
    }
    pred = predict_risk(high_risk_data, model)
    print(f"High risk (interactions): Risk = {pred}")
    
    print("\n" + "=" * 60)
    print("Training complete! Model saved to:", MODEL_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()
