import pickle
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv(r"C:\Users\Rahul\OneDrive\Documents\OPPO_RENO_10\Documents\Crop_Recommendation.csv")

# Split the data into features and labels
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
# predictions = model.predict(X_test)

pickle.dump(model, open("model.pkl", "wb"))
# Evaluate the model
# accuracy = model.score(X_test, y_test)
# print("Accuracy:", accuracy)
