# qss stylesheet
QSS_STYLE = """
/* Global Styles */
QMainWindow, QWidget {
    background-color: #f0f2f5; 
    font-family: "Segoe UI", Arial, Helvetica, sans-serif; 
    color: #333333; 
}

QLabel {
    font-size: 10pt;
    padding: 2px;
}

QLabel#TitleLabel {
    font-size: 18pt;
    font-weight: bold;
    color: #2c3e50; 
    padding-bottom: 10px;
    padding-top: 5px;
}

QLabel#SectionHeaderLabel {
    font-size: 12pt;
    font-weight: bold;
    color: #34495e; 
    margin-top: 10px;
    margin-bottom: 5px;
}

QLabel#CurrentPriceLabel {
    font-size: 15pt; 
    font-weight: bold;
    color: #16a085; 
    padding: 8px; 
    background-color: #e8f6f3;
    border-radius: 5px;
    border: 1px solid #d0e8e1;
}

QLineEdit, QTextEdit, QComboBox {
    background-color: #ffffff;
    border: 1px solid #d1d8e0; 
    border-radius: 5px;
    padding: 6px;
    font-size: 10pt;
    color: #333; 
}

QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
    border: 1px solid #3498db; 
}

QTextEdit {
    selection-background-color: #aed6f1;
    selection-color: #222;
}

QPushButton {
    background-color: #3498db; 
    color: white;
    font-size: 10pt;
    font-weight: bold;
    padding: 10px 18px;
    border-radius: 5px;
    border: none;
    min-height: 20px;
}

QPushButton:hover {
    background-color: #2980b9; 
}

QPushButton:pressed {
    background-color: #1f618d; 
}

QPushButton:disabled {
    background-color: #bdc3c7;
    color: #7f8c8d;
}

QTabWidget::pane {
    border: 1px solid #d1d8e0;
    border-top: none;
    background: #ffffff;
    border-bottom-left-radius: 5px;
    border-bottom-right-radius: 5px;
    padding: 10px;
}

QTabBar::tab {
    background: #e4e7eb;
    color: #566573;
    border: 1px solid #d1d8e0;
    border-bottom: none; 
    padding: 10px 20px;
    margin-right: 1px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}

QTabBar::tab:selected {
    background: #ffffff; 
    color: #3498db; 
    font-weight: bold;
    border-bottom-color: #ffffff; 
}

QTabBar::tab:hover:!selected {
    background: #eff1f3;
}

QProgressBar {
    border: 1px solid #d1d8e0;
    border-radius: 5px;
    text-align: center;
    color: #333333;
    font-size: 9pt;
    height: 22px;
}

QProgressBar::chunk {
    background-color: #27ae60;
    border-radius: 4px;
    margin: 1px;
}

QGroupBox {
    font-size: 11pt;
    font-weight: bold;
    color: #34495e;
    border: 1px solid #d1d8e0;
    border-radius: 5px;
    margin-top: 10px; 
    padding-top: 20px; 
    padding-bottom: 10px;
    padding-left: 10px;
    padding-right: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left; 
    padding: 0 5px 0 5px;
    left: 10px; 
    background-color: #f0f2f5; 
}

QMessageBox QLabel {
    font-size: 10pt;
    color: #333;
}

QMessageBox QPushButton {
    min-width: 80px;
}
"""