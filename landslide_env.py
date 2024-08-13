import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class LandslideEnv:
    def __init__(self, data_path):
        self.data_path = data_path  # Store the path as an instance variable
        self.state_size = None
        self.action_size = 2
        self.current_index = 0
        self.data, self.labels = self.load_data()  # Call load_data without arguments
        self.state_size = self.data.shape[1]
        self.reset()

    def load_data(self):  # Corrected method definition
        df = pd.read_csv(self.data_path)
        
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

        return X_transformed, y

    def reset(self):
        self.current_index = 0
        self.state = self.data[self.current_index]
        return self.state

    def step(self, action):
        done = False
        reward = 0
        
        if action == 1:  # Predict landslide
            if self.labels[self.current_index] == 1:
                reward = 10
            else:
                reward = -10
        else:  # Do not predict landslide
            if self.labels[self.current_index] == 0:
                reward = 5
            else:
                reward = -5
        
        self.current_index += 1
        if self.current_index >= len(self.data):
            done = True
        else:
            self.state = self.data[self.current_index]

        return self.state, reward, done, {}

    def render(self):
        print(f"Current State: {self.state}")

    def close(self):
        pass
