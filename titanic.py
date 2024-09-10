import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the Titanic dataset
data = pd.read_csv("Titanic-Dataset.csv")  # Ensure the Titanic dataset is in the same directory or provide the correct path

# Data preprocessing
# Drop columns that are not useful for prediction
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)  # Fill missing Age with median
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)  # Fill missing Embarked with mode

# Convert categorical variables into numerical values
label_encoders = {}
for column in ['Sex', 'Embarked']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target variable
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Train the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create the GUI
root = tk.Tk()
root.title("Titanic Survival Prediction")

# Function to predict survival
def predict_survival():
    pclass = int(pclass_entry.get())
    sex = sex_combo.get()
    age = float(age_entry.get())
    sibsp = int(sibsp_entry.get())
    parch = int(parch_entry.get())
    fare = float(fare_entry.get())
    embarked = embarked_combo.get()

    # Encode the input data
    sex_encoded = label_encoders['Sex'].transform([sex])[0]
    embarked_encoded = label_encoders['Embarked'].transform([embarked])[0]

    # Create a new passenger array with the encoded data
    new_passenger = [[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]]

    # Predict the survival
    predicted_survival = model.predict(new_passenger)

    # Display the predicted survival
    result_label.config(text=f"Predicted Survival: {'Survived' if predicted_survival[0] == 1 else 'Did not survive'}")

# Create GUI elements
tk.Label(root, text="Passenger Class (1, 2, or 3):").pack()
pclass_entry = tk.Entry(root)
pclass_entry.pack()

tk.Label(root, text="Sex (male or female):").pack()
sex_combo = ttk.Combobox(root, values=['male', 'female'])
sex_combo.pack()

tk.Label(root, text="Age:").pack()
age_entry = tk.Entry(root)
age_entry.pack()

tk.Label(root, text="Siblings/Spouses Aboard (SibSp):").pack()
sibsp_entry = tk.Entry(root)
sibsp_entry.pack()

tk.Label(root, text="Parents/Children Aboard (Parch):").pack()
parch_entry = tk.Entry(root)
parch_entry.pack()

tk.Label(root, text="Fare:").pack()
fare_entry = tk.Entry(root)
fare_entry.pack()

tk.Label(root, text="Embarked (C, Q, or S):").pack()
embarked_combo = ttk.Combobox(root, values=['C', 'Q', 'S'])
embarked_combo.pack()

predict_button = tk.Button(root, text="Predict Survival", command=predict_survival)
predict_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

# Run the GUI
root.mainloop()