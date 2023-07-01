import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import gradio as gr

# Load the dataset
dataset = pd.read_csv('Heart_disease_statlog.csv')  # Replace 'your_dataset.csv' with the actual filename

# Split the dataset into features and target
X = dataset.drop('target', axis=1)
y = dataset['target']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
model_path = "heart_disease_model.joblib"
joblib.dump(model, model_path)

# Define the prediction function
def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    loaded_model = joblib.load(model_path)
    prediction = loaded_model.predict(input_data)
    return prediction[0]

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Accuracy:", accuracy)
print("Mean Absolute Error:", mae)

# Create the Gradio interface
iface = gr.Interface(
    fn=predict, 
    inputs=["number", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number"], 
    outputs="text"
)

# Launch the Gradio interface with sharing enabled
iface.launch(share=True)
