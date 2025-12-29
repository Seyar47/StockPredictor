import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from gui import StockPredictorGUI 
from styles import QSS_STYLE      

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS_STYLE) 
    
    try:
        app.setWindowIcon(QIcon('stock.jpg')) 
    except Exception as e:
        print(f"Could not load app icon: {e}")

    window = StockPredictorGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()