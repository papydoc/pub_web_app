# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# 1-install the necessary libraries (if you haven't already):
 # 2-Create a Python file (e.g., iris_classifier_app.py) and import the necessary libraries:
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 3-Load the iris dataset using the datasets.load_iris() function and assign the data and target variables to X and Y, respectively:
iris_data = load_iris()
# Show the dataset's keys
print(list(iris_data))

# Description of the dataset
print(iris_data['DESCR'])

# Location of the CSV file containing the data being imported
print(iris_data['filename'])

X = iris_data.data
Y = iris_data.target

# 4-Set up a Random Forest Classifier and fit the model using the RandomForestClassifier() and fit() functions:
# Create the Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Fit the model
rf_classifier.fit(X, Y)

# 5-Create the Streamlit app:

def main():
    # Add a title and header to the app
    st.title("Iris Flower Type Prediction")
    st.header("Enter the values to predict the type of iris flower")

    # Add input fields for sepal length, sepal width, petal length, and petal width
    sepal_length = st.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    sepal_width = st.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
    petal_length = st.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
    petal_width = st.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

    # Define a prediction button
    if st.button("Predict"):
        # Create a feature vector from user inputs
        user_input = [[sepal_length, sepal_width, petal_length, petal_width]]

        # Use the classifier to predict the type of iris flower
        prediction = rf_classifier.predict(user_input)

        # Display the predicted type of iris flower
        st.write(f"Predicted Iris Flower Type: {iris_data.target_names[prediction[0]]}")


if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
