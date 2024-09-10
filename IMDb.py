import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("IMDb Movies India.csv", encoding='latin-1')

# Remove rows with missing ratings
data = data[~data['Rating'].isna()]

# Data preprocessing
# Handle categorical features using Label Encoding
label_encoders = {}
for column in ['Actor 1', 'Actor 2', 'Actor 3']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target variable
X = data[['Actor 1', 'Actor 2', 'Actor 3']]
y = data['Rating']

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Create the GUI
root = tk.Tk()
root.title("Movie Rating Prediction")

# Get unique actor names for the Combobox
actor1_names = label_encoders['Actor 1'].inverse_transform(range(len(label_encoders['Actor 1'].classes_)))
actor2_names = label_encoders['Actor 2'].inverse_transform(range(len(label_encoders['Actor 2'].classes_)))
actor3_names = label_encoders['Actor 3'].inverse_transform(range(len(label_encoders['Actor 3'].classes_)))

# Function to predict movie rating
def predict_rating():
    actor1 = actor1_combo.get()
    actor2 = actor2_combo.get()
    actor3 = actor3_combo.get()

    try:
        # Encode the input actors
        encoded_actors = [
            label_encoders['Actor 1'].transform([actor1])[0],
            label_encoders['Actor 2'].transform([actor2])[0],
            label_encoders['Actor 3'].transform([actor3])[0]
        ]

        # Create a new movie array with the encoded actors
        new_movie = [encoded_actors]

        # Predict the rating
        predicted_rating = model.predict(new_movie)

        # Display the predicted rating
        result_label.config(text=f"Predicted Rating: {predicted_rating[0]:.2f}")
    except ValueError as e:
        result_label.config(text=f"Error: {e}")

# Create the GUI elements
actor1_label = tk.Label(root, text="Actor 1:")
actor1_label.pack()
actor1_combo = ttk.Combobox(root, values=actor1_names)
actor1_combo.pack()

actor2_label = tk.Label(root, text="Actor 2:")
actor2_label.pack()
actor2_combo = ttk.Combobox(root, values=actor2_names)
actor2_combo.pack()

actor3_label = tk.Label(root, text="Actor 3:")
actor3_label.pack()
actor3_combo = ttk.Combobox(root, values=actor3_names)
actor3_combo.pack()

predict_button = tk.Button(root, text="Predict Rating", command=predict_rating)
predict_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

# Run the GUI
root.mainloop()