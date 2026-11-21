"""Application theme (modern dark glass). Change colours here only."""

ACCENT = "#62A7FF"
ACCENT_2 = "#9B7CF6"
PINK = "#F47AC8"
BG = "#101520"
PANEL = "#171C28"
CARD = "#202735"
BORDER = "#364157"
TEXT = "#F2F5FA"
MUTED = "#A4AEC0"

STYLESHEET = f"""
* {{ font-size: 10pt; color: {TEXT}; }}
QMainWindow {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #090D18, stop:0.38 #11172A, stop:0.72 #24172F,
                                stop:1 #171229);
}}
QWidget#central {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 rgba(14, 21, 38, 220),
                                stop:0.62 rgba(35, 25, 55, 205),
                                stop:1 rgba(58, 27, 66, 205));
}}
QToolBar {{
    background: rgba(45, 49, 74, 205);
    border: none;
    border-bottom: 1px solid rgba(244, 122, 200, 80);
    border-radius: 0 0 14px 14px;
    padding: 5px;
    spacing: 5px;
}}
QToolBar QToolButton {{ padding: 7px 11px; border-radius: 7px; font-weight: 600; }}
QToolBar QToolButton:hover {{
    background: rgba(244, 122, 200, 55);
    border: 1px solid rgba(244, 122, 200, 130);
}}
QStatusBar {{
    background: rgba(26, 29, 48, 210);
    color: {MUTED};
    border-top: 1px solid rgba(244, 122, 200, 65);
}}

QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{
    background: rgba(45, 42, 70, 190);
    border-bottom: 1px solid rgba(244, 122, 200, 45);
    padding: 8px;
    font-weight: 600;
}}
QFrame#sidebar, QWidget#dockBody {{
    background: rgba(28, 31, 51, 205);
    border-right: 1px solid rgba(244, 122, 200, 55);
}}

QTabWidget::pane {{ border: none; background: transparent; }}
QTabBar::tab {{
    background: transparent;
    color: {MUTED};
    padding: 11px 20px;
    margin: 4px 2px 0 2px;
    border: none;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
}}
QTabBar::tab:selected {{
    color: {TEXT};
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 rgba(98, 167, 255, 90),
                                stop:1 rgba(244, 122, 200, 75));
    border: 1px solid rgba(244, 188, 226, 65);
    border-bottom: 2px solid {PINK};
}}
QTabBar::tab:hover {{
    color: {TEXT};
    background: rgba(244, 122, 200, 42);
}}

QLabel#title {{ font-size: 14pt; font-weight: 700; }}
QLabel#owner {{ color: {ACCENT}; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}

QGroupBox {{
    background: rgba(53, 54, 83, 170);
    border: 1px solid rgba(244, 170, 218, 75);
    border-radius: 12px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; font-weight: 600; }}

QPushButton {{
    background: rgba(61, 61, 91, 210);
    border: 1px solid rgba(244, 170, 218, 65);
    border-radius: 7px;
    padding: 7px 12px;
}}
QPushButton:hover {{
    border-color: {PINK};
    background: rgba(244, 122, 200, 52);
}}
QPushButton:disabled {{ color: #5d6577; border-color: {BORDER}; }}
QPushButton#primary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #6B8DF5, stop:0.55 #A56DE8, stop:1 #F47AC8);
    border: 1px solid rgba(255, 215, 243, 90);
    color: white;
    font-weight: 700;
}}
QPushButton#primary:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                        stop:0 #7FA6FF, stop:0.55 #BD82F2,
                                                        stop:1 #FF92D4); }}
QPushButton#primary:disabled {{ background: #34507F; color: #b8c4d9; }}
QPushButton#segment {{ border-radius: 0; padding: 5px 14px; }}
QPushButton#segment:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}

QSpinBox, QDoubleSpinBox, QComboBox {{ background: rgba(20, 22, 42, 205); border: 1px solid rgba(244, 170, 218, 72); border-radius: 7px;
                                        padding: 4px 6px; min-height: 20px; }}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {PINK}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; selection-background-color: {ACCENT}; color: {TEXT}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}

QFrame#card {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 rgba(85, 87, 125, 210),
                                stop:1 rgba(70, 45, 91, 190));
    border: 1px solid rgba(255, 184, 226, 78);
    border-radius: 12px;
}}
QLabel#cardTitle {{ color: {MUTED}; font-size: 8.5pt; }}
QLabel#cardValue {{ font-size: 15pt; font-weight: 700; }}

QListWidget {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px; }}
QListWidget::item {{ padding: 6px; border-radius: 6px; }}
QListWidget::item:selected {{ background: #2F4570; }}

QTableWidget {{ background: {PANEL}; gridline-color: {BORDER}; border: 1px solid {BORDER}; border-radius: 6px;
               alternate-background-color: #262C38; }}
QHeaderView::section {{ background: {CARD}; padding: 6px; border: none; border-right: 1px solid {BORDER};
                       font-weight: 600; }}
QGraphicsView {{
    background: rgba(8, 10, 24, 220);
    border: 1px solid rgba(244, 170, 218, 82);
    border-radius: 12px;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget#scrollBody {{ background: transparent; }}
QScrollArea > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QProgressBar {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 5px; max-height: 12px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QToolTip {{ background: #F4F7FB; color: #111827; border: 1px solid #C8D1E0; padding: 6px; }}

/* Native-looking light dialogs stay readable over the dark application theme. */
QDialog, QMessageBox, QFileDialog {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #FFFFFF, stop:1 #EEF3FA);
    color: #111827;
}}
QDialog QLabel, QMessageBox QLabel, QFileDialog QLabel,
QDialog QCheckBox, QMessageBox QCheckBox, QFileDialog QCheckBox,
QDialog QRadioButton, QMessageBox QRadioButton, QFileDialog QRadioButton {{
    color: #111827;
}}
QDialog QPushButton, QMessageBox QPushButton, QFileDialog QPushButton {{
    background: #FFFFFF;
    color: #111827;
    border: 1px solid #C7D0DE;
    border-radius: 7px;
    padding: 7px 14px;
}}
QDialog QPushButton:hover, QMessageBox QPushButton:hover, QFileDialog QPushButton:hover {{
    background: #EAF2FF;
    color: #111827;
    border-color: #62A7FF;
}}
QDialog QLineEdit, QMessageBox QLineEdit, QFileDialog QLineEdit,
QDialog QComboBox, QMessageBox QComboBox, QFileDialog QComboBox,
QDialog QSpinBox, QMessageBox QSpinBox, QFileDialog QSpinBox,
QDialog QDoubleSpinBox, QMessageBox QDoubleSpinBox, QFileDialog QDoubleSpinBox {{
    background: #FFFFFF;
    color: #111827;
    border: 1px solid #C7D0DE;
    border-radius: 6px;
    padding: 5px 7px;
}}
QDialog QComboBox QAbstractItemView, QFileDialog QComboBox QAbstractItemView {{
    background: #FFFFFF;
    color: #111827;
    selection-background-color: #DCEAFF;
    selection-color: #111827;
}}
QDialog QListView, QMessageBox QListView, QFileDialog QListView,
QDialog QTreeView, QMessageBox QTreeView, QFileDialog QTreeView,
QDialog QTableView, QMessageBox QTableView, QFileDialog QTableView {{
    background: #FFFFFF;
    color: #111827;
    alternate-background-color: #F5F7FA;
}}
"""
