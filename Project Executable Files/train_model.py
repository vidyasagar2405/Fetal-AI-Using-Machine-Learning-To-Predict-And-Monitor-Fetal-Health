import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('fetal_health.csv')

# Define features and target variable
X = data.drop(columns=['fetal_health'])
y = data['fetal_health']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a pickle file
with open('fetal_health_model.pkl', 'wb') as file:
    pickle.dump(model, file)

