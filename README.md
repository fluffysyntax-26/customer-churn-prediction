# Customer Churn Prediction

This project uses an Artificial Neural Network (ANN) to predict customer churn based on historical data. It includes a Streamlit web application that allows users to input customer details and receive a real-time prediction on whether the customer is likely to churn.

## Features

  * **Customer Churn Prediction**: Predicts if a customer is likely to churn or not.
  * **Web Interface**: An interactive web application built with Streamlit for easy use.
  * **Trained Model**: Includes a pre-trained Keras model for immediate use.

-----

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You will need to have Python installed on your system. You can download it from the official [Python website](https://www.python.org/downloads/).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

    *Note: On Windows, the activation command is `venv\Scripts\activate`*

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

-----

## Usage

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will open a new tab in your web browser with the application running locally at `http://localhost:8501`.

-----

## Files in this Repository

  * **`app.py`**: The main file for the Streamlit web application.
  * **`nn.ipynb`**: A Jupyter Notebook containing the code for training the neural network.
  * **`prediction.ipynb`**: A Jupyter Notebook demonstrating how to make predictions with the trained model.
  * **`Churn_Modelling.csv`**: The dataset used for training the model.
  * **`model.keras`**: The pre-trained Keras model.
  * **`requirements.txt`**: A list of all the Python libraries required for this project.
  * **`gender_le.pkl`, `geography_ohe.pkl`, `scaler.pkl`**: The saved encoders and scaler used for data preprocessing.

-----

## Model Training

The model is an Artificial Neural Network (ANN) built with TensorFlow and Keras. The training process involves:

1.  **Data Preprocessing**: Loading the data, handling categorical features, and scaling the data.
2.  **Model Building**: Creating a sequential model with two hidden layers.
3.  **Training**: Training the model with the preprocessed data and saving it as `model.keras`.

## Acknowledgments

This project uses the "Churn Modelling" dataset, which is available on Kaggle.