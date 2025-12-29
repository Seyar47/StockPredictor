# 📈 Stock Price Predictor Pro

A desktop application that utilizes Machine Learning to analyze historical stock data and forecast future price trends. Built with **Python**, **PyQt5**, and **scikit-learn**.

![App Screenshot](screenshot.png)
![App Screenshot](screenshot2.png)

## 🚀 Overview
This tool allows users to input any stock ticker (e.g., AAPL, TSLA), fetches real-time historical data, and trains multiple machine learning models on the fly to predict stock movement for the next 7 days.

Unlike simple scripts, this is a fully responsive GUI application that utilizes **multithreading** to perform heavy data processing and model training in the background without freezing the user interface.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **GUI:** PyQt5 (Qt Designer)
* **Machine Learning:** scikit-learn (Random Forest, Linear Regression, KNN)
* **Data Analysis:** Pandas, NumPy
* **Data Source:** yfinance API
* **Visualization:** Matplotlib (integrated into Qt)

## ✨ Key Features
* **Real-Time Data Pipeline:** Fetches live data via `yfinance`, cleans missing values, and engineers features (Moving Averages, Daily Returns, Volatility).
* **Multi-Model Training:** Automatically trains and compares three different algorithms:
    * Random Forest Regressor
    * Linear Regression
    * K-Nearest Neighbors (KNN)
* **Asynchronous Processing:** Implements a `QThread` worker architecture to handle fetching and training, ensuring the GUI remains responsive.
* **Interactive Visualization:** Dynamic charts plotting historical data, model performance (Actual vs. Predicted), and future 7-day forecasts.

## ⚙️ Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Seyar47/StockPredictor.git](https://github.com/Seyar47/StockPredictor.git)
    cd StockPredictor
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**
    ```bash
    python main.py
    ```

## 🧠 How It Works
1.  **Data Ingestion:** The app downloads 5 years of historical data for the requested ticker.
2.  **Feature Engineering:** It calculates technical indicators:
    * 5, 10, and 20-day Moving Averages (MA)
    * Daily Returns & Volume trends
3.  **Model Training:** The data is split (80/20) and scaled (`StandardScaler`). The app trains the models and selects the best performer (defaulting to Random Forest for predictions).
4.  **Forecasting:** The model predicts the closing price for the next 7 days based on the most recent market data.

## 📂 Project Structure
* `main.py`: Entry point of the application.
* `gui.py`: Handles the PyQt5 interface, plotting, and threading logic.
* `backend.py`: Contains the `StockPredictor` class (Data fetching, ML logic).
* `styles.py`: Custom QSS stylesheets

## 🚀 Future Improvements
* Add Long Short-Term Memory (LSTM) models for better time-series accuracy.
* Implement "Sentiment Analysis"
* Save trained models to disk to reduce loading times.