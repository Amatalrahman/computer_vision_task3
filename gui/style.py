"""Application theme (modern dark glass). Change colours here only."""

ACCENT = "#62A7FF"
ACCENT_2 = "#9B7CF6"
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
                                stop:0 #0D1119, stop:0.55 #121827, stop:1 #171522);
}}
QWidget#central {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 rgba(20, 27, 42, 245), stop:1 rgba(26, 21, 40, 245));
}}
QToolBar {{
    background: rgba(29, 35, 49, 235);
    border: none;
    border-bottom: 1px solid rgba(116, 143, 185, 70);
    padding: 5px;
    spacing: 5px;
}}
QToolBar QToolButton {{ padding: 7px 11px; border-radius: 7px; font-weight: 600; }}
QToolBar QToolButton:hover {{ background: rgba(98, 167, 255, 55); border: 1px solid rgba(98, 167, 255, 110); }}
QStatusBar {{ background: rgba(23, 28, 40, 235); color: {MUTED}; border-top: 1px solid rgba(116, 143, 185, 55); }}

QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: rgba(29, 35, 49, 230); padding: 8px; font-weight: 600; }}
QFrame#sidebar, QWidget#dockBody {{
    background: rgba(25, 31, 44, 238);
    border-right: 1px solid rgba(116, 143, 185, 55);
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
    background: rgba(98, 167, 255, 28);
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover {{ color: {TEXT}; background: rgba(255, 255, 255, 18); }}

QLabel#title {{ font-size: 14pt; font-weight: 700; }}
QLabel#owner {{ color: {ACCENT}; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}

QGroupBox {{
    background: rgba(32, 39, 53, 190);
    border: 1px solid rgba(151, 171, 204, 42);
    border-radius: 10px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; font-weight: 600; }}

QPushButton {{
    background: rgba(43, 51, 68, 235);
    border: 1px solid rgba(151, 171, 204, 48);
    border-radius: 7px;
    padding: 7px 12px;
}}
QPushButton:hover {{ border-color: {ACCENT}; background: rgba(55, 70, 94, 235); }}
QPushButton:disabled {{ color: #5d6577; border-color: {BORDER}; }}
QPushButton#primary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #4F93F5, stop:1 #8C73EB);
    border: 1px solid rgba(255, 255, 255, 35);
    color: white;
    font-weight: 700;
}}
QPushButton#primary:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                        stop:0 #68A7FF, stop:1 #A48AFA); }}
QPushButton#primary:disabled {{ background: #34507F; color: #b8c4d9; }}
QPushButton#segment {{ border-radius: 0; padding: 5px 14px; }}
QPushButton#segment:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}

QSpinBox, QDoubleSpinBox, QComboBox {{ background: rgba(16, 21, 32, 225); border: 1px solid rgba(151, 171, 204, 58); border-radius: 6px;
                                        padding: 4px 6px; min-height: 20px; }}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; selection-background-color: {ACCENT}; color: {TEXT}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}

QFrame#card {{
    background: rgba(39, 47, 64, 205);
    border: 1px solid rgba(151, 171, 204, 48);
    border-radius: 10px;
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
    background: rgba(9, 13, 21, 235);
    border: 1px solid rgba(151, 171, 204, 48);
    border-radius: 10px;
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
