import sys
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QLineEdit, QPushButton, QTextEdit,
                             QComboBox, QProgressBar, QTabWidget, QMessageBox,
                             QGroupBox, QSizePolicy, QApplication)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QIcon

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt  # <--- ADD THIS LINE

# IMPORT YOUR OTHER FILES HERE
from backend import StockPredictor

class DataWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker
        self.predictor = StockPredictor()
    
    def run(self):
        try:
            self.progress.emit(f"Fetching stock data for {self.ticker}...")
            success, msg = self.predictor.fetch_data(self.ticker)
            if not success:
                self.error.emit(msg); return
            
            self.progress.emit("Preprocessing data...")
            success, msg = self.predictor.preprocess_data()
            if not success:
                self.error.emit(msg); return
            
            self.progress.emit("Training models...")
            results, _, _, train_msg = self.predictor.train_models() # X_test, y_test not used directly by GUI here
            self.progress.emit(train_msg) 
            if results is None: # big big failure in training
                self.error.emit(train_msg if train_msg else "Model training failed critically."); return

            self.progress.emit("Generating predictions...")
            # use the model determined as best or default during training
            predictions, pred_msg = self.predictor.predict_future(model_name=self.predictor.trained_model_name_for_prediction)
            if predictions is None:
                self.error.emit(pred_msg); return
            
            current_price = self.predictor.get_current_price(self.ticker)
            
            result_data = {
                'predictor': self.predictor, 'results': results, 'predictions': predictions,
                'current_price': current_price, 'ticker': self.ticker,
                'prediction_model_name': self.predictor.trained_model_name_for_prediction
            }
            self.finished.emit(result_data)
        except Exception as e:
            self.error.emit(f"Critical error in processing thread: {str(e)}")

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=8, dpi=100):
        try:
            plt.style.use("seaborn-v0_8-pastel") 
        except IOError:
            print("Matplotlib style 'seaborn-v0_8-pastel' not found. Using default.")
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor("white")
            
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def _setup_ax_style(self, ax, title):
        ax.set_title(title, fontsize=14, fontweight='bold', color="#2c3e50")
        ax.title.set_position([.5, 1.05]) 
        ax.set_xlabel(ax.get_xlabel(), fontsize=11, color="#444")
        ax.set_ylabel(ax.get_ylabel(), fontsize=11, color="#444")
        ax.tick_params(axis='x', colors='#555', labelsize=9)
        ax.tick_params(axis='y', colors='#555', labelsize=9)
        ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc')
        if ax.legend_ != None: 
            ax.legend(fontsize=9, frameon=True, facecolor='#fdfefe', edgecolor='#d1d8e0')


    def plot_stock_history(self, data, ticker):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(data.index, data['Close'], label='Close Price', linewidth=1.5, color="#007ACC")
        ax.plot(data.index, data['MA_5'], label='5-day MA', alpha=0.7, linestyle='--', color="#F08A5D")
        ax.plot(data.index, data['MA_10'], label='10-day MA', alpha=0.7, linestyle='--', color="#B83B5E")
        ax.plot(data.index, data['MA_20'], label='20-day MA', alpha=0.7, linestyle='--', color="#6A2C70")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        self._setup_ax_style(ax, f'{ticker} Stock Price History')
        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_predictions(self, actual, predicted, model_name):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.scatter(actual, predicted, alpha=0.6, edgecolors='#555', color="#5DADE2", s=30)
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=1.5, label="Ideal Fit")
        ax.set_xlabel('Actual Price ($)')
        ax.set_ylabel('Predicted Price ($)')
        self._setup_ax_style(ax, f'{model_name}: Actual vs Predicted')
        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_future_predictions(self, predictions, current_price, ticker):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        days_axis = range(1, len(predictions) + 1)

        plot_label = "Predicted Prices"
        line_color = "#FF6F61" 
        marker_color = "#FF6F61"

        if current_price is not None:
            ax.plot([0] + list(days_axis), [current_price] + predictions, 'o-', linewidth=2, markersize=6, label=plot_label, color=line_color, markerfacecolor=marker_color)
            ax.axhline(y=current_price, color='#27AE60', linestyle=':', alpha=0.8, label=f'Current: ${current_price:.2f}')
            ax.annotate(f'${current_price:.2f}', (0, current_price), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, color="green", weight='bold')
        else:
            ax.plot(list(days_axis), predictions, 'o-', linewidth=2, markersize=6, label=plot_label, color=line_color, markerfacecolor=marker_color)
            ax.text(0.05, 0.95, "Current Price: N/A", transform=ax.transAxes, fontsize=10, va='top', color='gray')

        ax.set_xlabel('Days Ahead')
        ax.set_ylabel('Predicted Price ($)')
        self._setup_ax_style(ax, f'{ticker} - {len(predictions)}-Day Price Forecast')
        
        for i, pred_val in enumerate(predictions):
            day_x = days_axis[i]
            ax.annotate(f'${pred_val:.2f}', (day_x, pred_val), textcoords="offset points", 
                        xytext=(0, -15 if i % 2 == 0 else 10), ha='center', fontsize=8, color="#4A4A4A",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f2f5", ec="none", alpha=0.7)) 
        
        self.fig.tight_layout(pad=2.0)
        self.draw()

class StockPredictorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.results = None
        self.last_analysis_data = None 
        self.initUI()

    def initUI(self):
        self.setWindowTitle('🔮 Stock Price Predictor')
        self.setGeometry(50, 50, 1280, 850) 

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15) 
        main_layout.setSpacing(10) 

        title = QLabel('🔮 Stock Price Predictor')
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        input_group_box = QGroupBox("Stock Analyzer")
        input_group_layout = QHBoxLayout()
        input_group_layout.addWidget(QLabel('Ticker:'))
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText('e.g., AAPL, TSLA, MSFT')
        input_group_layout.addWidget(self.ticker_input, 1)
        self.analyze_btn = QPushButton('🚀 Analyze')
        self.analyze_btn.clicked.connect(self.analyze_stock)
        input_group_layout.addWidget(self.analyze_btn)
        input_group_box.setLayout(input_group_layout)
        main_layout.addWidget(input_group_box)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel('Enter a stock ticker and click Analyze.')
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        self.results_tab = QWidget()
        self.tab_widget.addTab(self.results_tab, '📊 Results & Predictions')
        
        self.charts_tab = QWidget()
        self.tab_widget.addTab(self.charts_tab, '📈 Charts')
        
        self.setup_results_tab()
        self.setup_charts_tab()

    def setup_results_tab(self):
        layout = QVBoxLayout(self.results_tab)
        layout.setSpacing(15)

        # current price Section
        price_group = QGroupBox("Real-time Data")
        price_layout = QVBoxLayout()
        self.current_price_label = QLabel('Current Price: N/A')
        self.current_price_label.setObjectName("CurrentPriceLabel")
        self.current_price_label.setAlignment(Qt.AlignCenter)
        price_layout.addWidget(self.current_price_label)
        price_group.setLayout(price_layout)
        layout.addWidget(price_group)

        #  performance model Section
        perf_group = QGroupBox("Model Performance")
        perf_layout = QVBoxLayout()
        self.performance_text = QTextEdit()
        self.performance_text.setReadOnly(True)
        self.performance_text.setFixedHeight(180) 
        perf_layout.addWidget(self.performance_text)
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # predictions Section
        pred_group = QGroupBox("Price Forecast (7-Day)")
        pred_layout = QVBoxLayout()
        self.predictions_text = QTextEdit()
        self.predictions_text.setReadOnly(True)
        self.predictions_text.setFixedHeight(180) 
        pred_layout.addWidget(self.predictions_text)
        pred_group.setLayout(pred_layout)
        layout.addWidget(pred_group)

        layout.addStretch(1) 

    def setup_charts_tab(self):
        layout = QVBoxLayout(self.charts_tab)
        layout.setSpacing(10)
        
        chart_selection_layout = QHBoxLayout()
        chart_selection_layout.addWidget(QLabel('Select Chart:'))
        self.chart_combo = QComboBox()
        self.chart_combo.addItems(['Stock History', 'Model Performance (Random Forest)', 'Future Price Forecast'])
        self.chart_combo.currentTextChanged.connect(self.update_chart)
        chart_selection_layout.addWidget(self.chart_combo, 1)
        layout.addLayout(chart_selection_layout)
        
        self.plot_canvas = PlotCanvas(self)
        layout.addWidget(self.plot_canvas)

    def analyze_stock(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, 'Input Required', 'Please enter a stock ticker symbol.')
            return
        
        self.analyze_btn.setEnabled(False)
        self.status_label.setText(f"Starting analysis for {ticker}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.worker = DataWorker(ticker)
        self.worker.progress.connect(self.update_status)
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def update_status(self, message):
        self.status_label.setText(message)

    def on_analysis_complete(self, data):
        self.last_analysis_data = data
        self.predictor = data['predictor']
        self.results = data['results']
        current_price = data['current_price']
        predictions = data['predictions']
        ticker = data['ticker']
        pred_model_name = data['prediction_model_name']

        if current_price is not None:
            self.current_price_label.setText(f'Current Price ({ticker}): ${current_price:.2f}')
        else:
            self.current_price_label.setText(f'Current Price ({ticker}): N/A')
        
        perf_text = ""
        if self.results:
            for name, result in self.results.items():
                if result.get('error'):
                    perf_text += f"{name}:\n  Error: {result['error']}\n\n"
                else:
                    perf_text += f"{name}:\n"
                    perf_text += f"  Mean Squared Error: {result['mse']:.4f}\n"
                    perf_text += f"  R² Score: {result['r2']:.4f}\n\n"
        else:
            perf_text = "Model performance data not available."
        self.performance_text.setText(perf_text.strip())
        
        if predictions:
            pred_text = f"Forecast using {pred_model_name} model:\n\n"
            for i, pred in enumerate(predictions, 1):
                pred_text += f"Day {i}: ${pred:.2f}\n"
        else:
            pred_text = "Price forecast not available."
        self.predictions_text.setText(pred_text.strip())
        
        self.update_chart()
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0,100)
        self.status_label.setText(f'Analysis complete for {ticker}. Ready for new analysis or chart selection.')
        QMessageBox.information(self, "Analysis Complete", f"Successfully analyzed {ticker}.")


    def on_analysis_error(self, error_msg):
        QMessageBox.critical(self, 'Analysis Error', error_msg)
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText('Analysis failed. Please try again or check ticker.')

    def update_chart(self):
        if not self.last_analysis_data or not self.predictor or not self.results:
            self.plot_canvas.fig.clear()
            ax = self.plot_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data to display.\nPlease analyze a stock first.", 
                      ha='center', va='center', fontsize=12, color="#777")
            self.plot_canvas.draw()
            return
        
        chart_type = self.chart_combo.currentText()
        ticker = self.last_analysis_data['ticker']
        
        self.plot_canvas.fig.clear() 

        if chart_type == 'Stock History':
            if self.predictor.data is not None and not self.predictor.data.empty:
                self.plot_canvas.plot_stock_history(self.predictor.data, ticker)
            else:
                ax = self.plot_canvas.fig.add_subplot(111)
                ax.text(0.5, 0.5, "No historical data to display.", ha='center', va='center')
                self.plot_canvas.draw()
        elif chart_type == 'Model Performance (Random Forest)': 
            model_key = 'Random Forest'
            if model_key in self.results and \
               isinstance(self.results[model_key].get('actual'), np.ndarray) and \
               isinstance(self.results[model_key].get('predictions'), np.ndarray):
                rf_result = self.results[model_key]
                self.plot_canvas.plot_predictions(rf_result['actual'], rf_result['predictions'], model_key)
            else:
                ax = self.plot_canvas.fig.add_subplot(111)
                ax.text(0.5, 0.5, f"{model_key} performance data not available.", ha='center', va='center')
                self.plot_canvas.draw()
        elif chart_type == 'Future Price Forecast':
            predictions = self.last_analysis_data['predictions']
            current_price = self.last_analysis_data['current_price']
            if predictions is not None:
                self.plot_canvas.plot_future_predictions(predictions, current_price, ticker)
            else:
                ax = self.plot_canvas.fig.add_subplot(111)
                ax.text(0.5, 0.5, "Future price forecast not available.", ha='center', va='center')
                self.plot_canvas.draw()
        else:
            self.plot_canvas.draw()