"""Model training script with hyperparameter tuning."""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import json

def create_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(random_state=42))
    ])

def train(X_train, y_train, param_grid=None):
    if param_grid is None:
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.2]
        }
    
    pipeline = create_pipeline()
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV R2 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def save_model(model, path='model.pkl'):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

if __name__ == '__main__':
    print("Model training module ready. Import and call train() with your data.")
