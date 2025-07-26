import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load data
data = pd.read_csv('data.csv')

# Example: Features and Target
X = data[['experience', 'test_score', 'interview_score']]
y = data['salary']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('salary_model.pkl', 'wb') as file:
    pickle.dump(model, file)
