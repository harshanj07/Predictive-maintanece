from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from joblib import dump

# Generate a sample dataset (Replace with your actual data)
X_train, y_train = make_classification(n_samples=100, n_features=6, random_state=42)

# Train the model
rfc = DecisionTreeClassifier()
rfc.fit(X_train, y_train)

# Save the model to 'model.joblib'
dump(rfc, 'model.joblib')
print("Model saved successfully.")
