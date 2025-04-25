# Diamond Price Prediction Project

This project uses machine learning to predict the price of a diamond based on various features like carat, depth, table, dimensions, cut, color, and clarity.

## Features

Carat, Depth, Table, x, y, z: Physical attributes of the diamond.

Cut, Color, Clarity: Quality characteristics that affect the price.

## Project Structure

diamond-price-prediction/
│
├── frontend/ # React UI for user input and prediction display
│ └── DiamondPricePredictor.js
│
├── backend/ # FastAPI server for serving the ML model
│ ├── app.py # FastAPI app for prediction API
│ └── diamond_price_model.pkl # Trained machine learning model
│
├── data/ # Dataset (for training the model)
│ └── diamonds.csv # Sample dataset
│
├── requirements.txt # Backend Python dependencies
└── README.md # Project documentation

## How It Works

Backend: The FastAPI app serves the trained machine learning model at the /predict endpoint. It processes diamond feature data and predicts the price.

Frontend: React app collects user input, sends it to the backend API, and displays the predicted diamond price.

## Machine Learning Model

Model: The diamond price prediction model is trained using a dataset of diamonds and saved as diamond_price_model.pkl.

Training: The model is trained using features like carat, depth, and clarity. The model is then serialized and served via the FastAPI backend.
