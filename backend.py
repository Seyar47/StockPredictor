import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scaler = StandardScaler()
        self.trained_model_name_for_prediction = 'Random Forest' # defualt model for predictions

    def fetch_data(self, ticker, period="5y"):
        try:
            stock = yf.Ticker(ticker)
            self.data = stock.history(period=period)
            if self.data.empty:
                return False, f"no data found for ticker {ticker} with period {period}."
            return True, f"Successfully fetched {len(self.data)} days of data for {ticker}"
        except Exception as e:
            return False, f"Error fetching data for {ticker}: {str(e)}"

    def preprocess_data(self):
        if self.data is None or self.data.empty:
            return False, "No data available to preprocess."
        
        try:
            self.data = self.data.dropna(subset=['Close'])
            if self.data.empty:
                return False, "No valid historical data (e.g., 'Close' price missing) after initial NaN drop."
            
            self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
            self.data['MA_10'] = self.data['Close'].rolling(window=10).mean()
            self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
            self.data['Prev_Close'] = self.data['Close'].shift(1)
            self.data['Daily_Return'] = self.data['Close'].pct_change()
            self.data['Price_Change'] = self.data['Close'] - self.data['Prev_Close']
            self.data['Volume_MA'] = self.data['Volume'].rolling(window=5).mean()
            self.data['Target'] = self.data['Close'].shift(-1)
            self.data = self.data.dropna()
            
            if self.data.empty:
                return False, "Not enough data to create features (e.g., too short history for MAs or target)."
            
            return True, "Data preprocessing completed successfully"
        except Exception as e:
            return False, f"Error in preprocessing: {str(e)}"

    def prepare_features(self):
        feature_columns = ['Close', 'MA_5', 'MA_10', 'MA_20', 'Prev_Close', 
                           'Daily_Return', 'Price_Change', 'Volume', 'Volume_MA']
        X = self.data[feature_columns].values
        y = self.data['Target'].values
        return X, y

    def train_models(self):
        if self.data is None or self.data.empty:
            return None, None, None, "Cannot train models: No preprocessed data available."

        X, y = self.prepare_features()
        if X.shape[0] < 2: # we need at least 2 samples to split
             return None, None, None, "Cannot train models: Not enough data after feature preparation."

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        if len(X_train) == 0 or len(X_test) == 0 :
            return None, None, None, "Cannot train models: Data split resulted in empty train/test set."

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models_init = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        
        results = {}
        self.models = {} # previous models needs to be cleared
        
        for name, model_instance in models_init.items():
            try:
                if name == 'Linear Regression' or name == 'KNN':
                    model_instance.fit(X_train_scaled, y_train)
                    y_pred = model_instance.predict(X_test_scaled)
                else:
                    model_instance.fit(X_train, y_train)
                    y_pred = model_instance.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {'model': model_instance, 'mse': mse, 'r2': r2, 'predictions': y_pred, 'actual': y_test}
                self.models[name] = model_instance # store the trained model
            except Exception as e:
                results[name] = {'model': None, 'mse': float('inf'), 'r2': float('-inf'), 'predictions': None, 'actual': y_test, 'error': str(e)}
        
        self.trained_model_name_for_prediction = 'Random Forest' 
        if 'Random Forest' not in self.models or self.models['Random Forest'] is None:
            # Fallback if RF failed
            best_model_name = None
            best_r2 = -float('inf')
            for name, res in results.items():
                if res['model'] is not None and res['r2'] > best_r2:
                    best_r2 = res['r2']
                    best_model_name = name
            if best_model_name:
                 self.trained_model_name_for_prediction = best_model_name

        return results, X_test, y_test, "Model training completed."

    def predict_future(self, days=7, model_name=None):
        if model_name is None:
            model_name = self.trained_model_name_for_prediction

        if model_name not in self.models or self.models[model_name] is None:
            return None, f"Model '{model_name}' not found or not trained successfully."
        
        model = self.models[model_name]
        
        feature_columns = ['Close', 'MA_5', 'MA_10', 'MA_20', 'Prev_Close', 
                           'Daily_Return', 'Price_Change', 'Volume', 'Volume_MA']
        
        if self.data is None or self.data.empty:
            return None, "No data available for prediction base."

        try:
            last_features_df = self.data[feature_columns].iloc[-1:].copy()
        except IndexError:
             return None, "Not enough historical data for prediction base."

        predictions = []
        current_features_np = last_features_df.values.copy()
        
        idx_close = feature_columns.index('Close')
        idx_prev_close = feature_columns.index('Prev_Close')
        idx_daily_return = feature_columns.index('Daily_Return')
        idx_price_change = feature_columns.index('Price_Change')

        for _ in range(days):
            if current_features_np.ndim == 1:
                current_features_np = current_features_np.reshape(1, -1)

            if model_name == 'Linear Regression' or model_name == 'KNN':
                scaled_features = self.scaler.transform(current_features_np)
                pred = model.predict(scaled_features)[0]
            else:
                pred = model.predict(current_features_np)[0]
            predictions.append(pred)
            
            close_of_day_d = current_features_np[0, idx_close]
            current_features_np[0, idx_close] = pred
            current_features_np[0, idx_prev_close] = close_of_day_d
            
            if close_of_day_d != 0:
                current_features_np[0, idx_daily_return] = (pred - close_of_day_d) / close_of_day_d
            else:
                current_features_np[0, idx_daily_return] = 0.0
            current_features_np[0, idx_price_change] = pred - close_of_day_d
            
        return predictions, "Predictions generated successfully"

    def get_current_price(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            current_data = stock.history(period="1d")
            if not current_data.empty:
                return current_data['Close'].iloc[-1]
            else:
                hist_short = stock.history(period="5d") 
                if not hist_short.empty:
                    return hist_short['Close'].iloc[-1]
                return None 
        except Exception as e:
            print(f"Error in get_current_price for {ticker}: {e}") 
            return None