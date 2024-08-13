import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Separate features and labels
    X = df.drop('Frequency of Landslides', axis=1)  # Assuming this is the target variable
    y = df['Frequency of Landslides']

    # Identify categorical and numerical columns
    categorical_features = ['Soil Type', 'Rock Type', 'Land Cover Type']
    binary_features = ['Fault Lines', 'Previous Landslide', 'Construction Activity', 'Deforestation']
    numerical_features = [col for col in X.columns if col not in categorical_features + binary_features + ['ID']]

    # Create preprocessing pipelines for each feature type
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('bin', 'passthrough', binary_features)
        ],
        remainder='drop'  # This will drop the 'ID' column
    )

    # Fit and transform the features
    X_transformed = preprocessor.fit_transform(X)

    # Convert to DataFrame for easier handling (optional)
    feature_names = (numerical_features +
                     [f"{feat}_{cat}" for feat in categorical_features for cat in preprocessor.named_transformers_['cat'].categories_[categorical_features.index(feat)]] +
                     binary_features)
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    return X_transformed_df, y

# Usage
X, y = load_data("your_data_file.csv")