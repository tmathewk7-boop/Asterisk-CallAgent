"""
caller_dashboard1.py — VerityLink AI Dashboard (Fixed Attribute Error)
"""
import pymysql
import hashlib
import sys
import zipfile
import json
import threading
import os
from typing import Dict, List
import requests
from datetime import datetime
import platform
import time

# Optional timezone
try:
    from tzlocal import get_localzone
    LOCAL_TZ = get_localzone()
except Exception:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("America/Chicago")

from PyQt6.QtGui import QPixmap, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QTableView,
    QHeaderView,
    QAbstractItemView,
    QFrame,
    QToolButton,
    QDialog,
    QTextEdit,
    QCheckBox,
    QStackedWidget,
    QLineEdit,
    QMessageBox,
    QScrollBar
)
from PyQt6.QtCore import (
    Qt,
    QTimer,
    QThread,
    pyqtSignal,
    QAbstractTableModel,
    QPropertyAnimation,
    QEasingCurve,
    QObject
)

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL = "https://asterisk-callagent-1.onrender.com"

# API Endpoints
DEFAULT_API_URL = f"{BASE_URL}/api/calls"
GET_CALLS_URL = f"{BASE_URL}/api/calls"
SETTINGS_URL = f"{BASE_URL}/api/settings"
STATUS_URL = f"{BASE_URL}/status"
TOGGLE_URL = f"{BASE_URL}/toggle"
SET_TIME_URL = f"{BASE_URL}/set-time-range"
REGISTER_URL = f"{BASE_URL}/register"
SERVER_VERSION_URL = f"{BASE_URL}/version.json"
DELETE_URL = f"{BASE_URL}/api/calls/delete"

AUTO_REFRESH_MS = 10_000
STATUS_POLL_MS = 10_000
HTTP_TIMEOUT = 30
CONFIG_PATH = os.path.expanduser("~/.caller_dashboard_config.json")
SETTINGS_FILE = "agent_settings.json"
CURRENT_VERSION = "1.0.2" 

VERSION_URL = "https://github.com/tmathewk7-boop/Asterisk-CallAgent/edit/main/update.zip/version.json"
UPDATE_ZIP_URL = "https://github.com/tmathewk7-boop/Asterisk-CallAgent/edit/main/update.zip"

# -----------------------------
# THEME COLORS
# -----------------------------
BASE_BLACK = "#0a0a0a"
PANEL_BLACK = "#121212"
BORDER_GRAY = "#2a2a2a"
TEXT_MAIN = "#e0e0e0"
TEXT_SUBTLE = "#a0a0a0"
ACCENT_BLUE = "#357DED"
ACCENT_RED = "#b22222"
ACCENT_GREEN = "#2e8b57"

# -----------------------------
# DATABASE FUNCTIONS
# -----------------------------
def get_db_connection():
    try:
        conn = pymysql.connect(
            host="40.233.108.163",
            user="mthom",
            password="Thomasm@27",
            database="veritylink",
            connect_timeout=5,
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except pymysql.err.OperationalError as e:
        print(f"Connection error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(name, username, password, access_key):
    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed."
    try:
        with conn.cursor() as cursor:
            sql_check = "SELECT id, phone_number, is_used FROM registration_keys WHERE access_key=%s"
            cursor.execute(sql_check, (access_key,))
            key_data = cursor.fetchone()

            if not key_data:
                return False, "Invalid License Key."
            if key_data['is_used']:
                return False, "This License Key has already been used."

            phone_number = key_data['phone_number']
            hashed = hash_password(password)
            sql_insert = "INSERT INTO users (name, username, password_hash, phone_number) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql_insert, (name, username, hashed, phone_number))

            sql_update = "UPDATE registration_keys SET is_used=1 WHERE id=%s"
            cursor.execute(sql_update, (key_data['id'],))

        conn.commit()
        return True, f"Account created successfully!\nAssigned Phone: {phone_number}"
    except pymysql.err.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        conn.close()

def login_user(username, password):
    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed."
    try:
        with conn.cursor() as cursor:
            hashed = hash_password(password)
            sql = """
                SELECT u.*, r.access_key as license_key 
                FROM users u 
                LEFT JOIN registration_keys r ON u.phone_number = r.phone_number 
                WHERE u.username=%s AND u.password_hash=%s
                LIMIT 1
            """
            cursor.execute(sql, (username, hashed))
            user = cursor.fetchone()
            
            if user:
                return True, user
            else:
                return False, "Invalid username or password."
    finally:
        conn.close()

def resource_path(filename: str):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(__file__), filename)

# -----------------------------
# VERSION & UPDATES
# -----------------------------
def fetch_version_info():
    try:
        r = requests.get(f"{VERSION_URL}?t={int(time.time())}", timeout=5)
        if r.status_code == 200:
            try:
                data = r.json()
                return data
            except json.JSONDecodeError:
                return None
        return None
    except Exception as e:
        print("Error fetching version info:", e)
        return None

def check_for_updates(parent=None):
    data = fetch_version_info()
    if not data:
        return False

    # Ensure this key matches your version.json file structure
    latest_version = data.get("version", CURRENT_VERSION)
    if latest_version != CURRENT_VERSION:
        changelog = data.get("changelog", "")
        dl_url = data.get("download_url", UPDATE_ZIP_URL)
        
        reply = QMessageBox.question(
            parent,
            "Update Available",
            f"A new version ({latest_version}) is available!\n\nChangelog:\n{changelog}\n\nDo you want to download and install it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            download_and_install(dl_url)
        return True
    return False

def download_and_install(url):
    try:
        local_zip = "update.zip"
        r = requests.get(url, stream=True)
        with open(local_zip, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        with zipfile.ZipFile(local_zip, "r") as zip_ref:
            zip_ref.extractall(os.getcwd())
        
        os.remove(local_zip)
        QMessageBox.information(None, "Update Complete", "Update installed. The application will now close. Please restart it.")
        sys.exit(0)
    except Exception as e:
        QMessageBox.warning(None, "Update Failed", f"Failed to update: {e}")

# -----------------------------
# TERMS & ACCEPTANCE
# -----------------------------
def has_accepted_terms():
    if not os.path.exists(CONFIG_PATH):
        return False
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f).get("accepted_terms", False)
    except:
        return False

def save_terms_acceptance():
    with open(CONFIG_PATH, "w") as f:
        json.dump({"accepted_terms": True}, f)

class TermsDialog(QDialog):
    def __init__(self,  parent=None):
        super().__init__(parent)
        self.setWindowTitle("Terms and Conditions")
        self.setFixedSize(600, 500)
        self.setStyleSheet(f"background-color: {PANEL_BLACK}; color: {TEXT_MAIN};")

        layout = QVBoxLayout(self)
        title = QLabel("Terms & Conditions")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT_MAIN}; margin-bottom: 10px;")
        layout.addWidget(title)

        terms_text = QTextEdit()
        terms_text.setReadOnly(True)
        terms_text.setStyleSheet(f"""
            background-color: {BASE_BLACK};
            color: {TEXT_SUBTLE};
            border: 1px solid {BORDER_GRAY};
            border-radius: 8px;
            padding: 15px;
            font-size: 11px;
        """)
        
        draft_terms = (
            "**1. Acceptance and License**\n"
            "By using this software, you agree to these terms. VerityLink Communications grants you a limited, non-exclusive, non-transferable license to use the software for its intended purpose only.\n\n"
            "**2. User Responsibilities and Compliance**\n"
            "• You warrant that you are authorized to use and monitor all call and data interactions processed by the App.\n"
            "• **Legal Compliance:** You are solely responsible for complying with all local, state, and federal laws regarding call recording, privacy, and data protection (e.g., TCPA, GDPR, CCPA). **VerityLink Communications is not responsible for your compliance failures.**\n\n"
            "**3. AI Output Disclaimer**\n"
            "• **Fallibility:** You acknowledge that the App utilizes Artificial Intelligence, and its outputs, summaries, or actions are machine-generated and may contain errors.\n"
            "• **Verification:** You agree that AI output is **not a substitute for human review**.\n\n"
            "**4. Limitation of Liability**\n"
            "**THE APP IS PROVIDED 'AS IS' WITHOUT ANY WARRANTIES.**\n"
        )
        
        terms_text.setText(draft_terms)
        layout.addWidget(terms_text)

        btn = QPushButton("Close")
        btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                color: #121212;
                font-weight: bold;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #b0b0b0;
            }
        """)
        btn.clicked.connect(self.accept_and_save)
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignRight)

    def accept_and_save(self):
        save_terms_acceptance()
        self.accept()

# -----------------------------
# AUTH DIALOGS
# -----------------------------
class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.user_data = None
        self.setWindowTitle("Login")
        self.setFixedSize(360, 250)
        self.setStyleSheet("background-color: #121212; color: #e0e0e0;")

        layout = QVBoxLayout(self)
        title = QLabel("Welcome Back")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        self.username = QLineEdit()
        self.username.setPlaceholderText("Username")
        self.password = QLineEdit()
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.EchoMode.Password)

        for w in [self.username, self.password]:
            w.setStyleSheet("background-color: #1e1e1e; padding: 8px; border-radius: 6px;")
            layout.addWidget(w)

        login_btn = QPushButton("Login")
        login_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                color: #121212;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #b0b0b0;
            }
        """)
        login_btn.clicked.connect(self.handle_login)
        layout.addWidget(login_btn)

        self.register_link = QPushButton("Create an account")
        self.register_link.setFlat(True)
        self.register_link.setStyleSheet("""
            QPushButton {
                color: #4da6ff; 
                text-decoration: underline; 
                background: transparent; 
                border: none;
            }
            QPushButton:hover {
                color: #74b3ff;
            }
        """)
        self.register_link.clicked.connect(self.open_register)
        layout.addWidget(self.register_link, alignment=Qt.AlignmentFlag.AlignCenter)

    def handle_login(self):
        username = self.username.text().strip()
        password = self.password.text().strip()
        if not username or not password:
            QMessageBox.warning(self, "Missing Info", "Please enter both username and password.")
            return

        success, data = login_user(username, password)
        if success:
            self.user_data = data if isinstance(data, dict) else {"name": username, "id": None}
            self.accept()
        else:
            QMessageBox.warning(self, "Login Failed", data)

    def open_register(self):
        reg_dialog = RegisterDialog()
        reg_dialog.exec()


class RegisterDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Create Account")
        self.setFixedSize(360, 450)
        self.setStyleSheet("background-color: #121212; color: #e0e0e0;")

        layout = QVBoxLayout(self)
        title = QLabel("Create a New Account")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        self.name = QLineEdit()
        self.name.setPlaceholderText("Full Name")
        
        self.username = QLineEdit()
        self.username.setPlaceholderText("Username")
        
        self.password = QLineEdit()
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.license_key = QLineEdit()
        self.license_key.setPlaceholderText("License Key (Provided by Admin)")

        for w in [self.name, self.username, self.password, self.license_key]:
            w.setStyleSheet("background-color: #1e1e1e; padding: 8px; border-radius: 6px;")
            layout.addWidget(w)

        terms_layout = QHBoxLayout()
        self.terms_checkbox = QCheckBox("I Agree to the Terms and Conditions")
        self.terms_checkbox.setStyleSheet("color: #ccc; font-size: 13px;")
        self.view_terms_btn = QPushButton("View")
        self.view_terms_btn.setFlat(True)
        self.view_terms_btn.setStyleSheet("""
            QPushButton {
                color: #4da6ff; 
                text-decoration: underline; 
                background: transparent; 
                border: none;
            }
            QPushButton:hover {
                color: #74b3ff;
            }
        """)
        self.view_terms_btn.clicked.connect(self.show_terms_dialog)
        terms_layout.addWidget(self.terms_checkbox)
        terms_layout.addStretch()
        terms_layout.addWidget(self.view_terms_btn)
        layout.addLayout(terms_layout)

        self.register_btn = QPushButton("Register")
        self.register_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                color: #121212;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #b0b0b0;
            }
        """)
        self.register_btn.clicked.connect(self.handle_register)
        layout.addWidget(self.register_btn)

    def show_terms_dialog(self):
        dlg = TermsDialog()
        dlg.exec()

    def handle_register(self):
        name = self.name.text().strip()
        username = self.username.text().strip()
        password = self.password.text().strip()
        key = self.license_key.text().strip()

        if not all([name, username, password, key]):
            QMessageBox.warning(self, "Missing Info", "Please fill all fields, including License Key.")
            return
        if not self.terms_checkbox.isChecked():
            QMessageBox.warning(self, "Agreement Required", "You must agree to the Terms and Conditions.")
            return

        success, msg = register_user(name, username, password, key)
        if success:
            QMessageBox.information(self, "Success", msg)
            self.accept()
        else:
            QMessageBox.warning(self, "Failed", msg)


# -----------------------------
# STARTUP SCREEN
# -----------------------------
class StartupScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet(f"background-color: {BASE_BLACK}; color: {TEXT_MAIN};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_path = resource_path("logo6.png")
        pix = QPixmap(logo_path)

        if pix.isNull():
            logo = QLabel("VerityLink")
            logo.setFont(QFont("Segoe UI", 52, QFont.Weight.Bold))
        else:
            logo = QLabel()
            logo.setPixmap(
                pix.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )

        logo.setFixedSize(400, 400)
        layout.addWidget(logo, alignment=Qt.AlignmentFlag.AlignCenter)

        self.loading = QLabel("Initializing...")
        self.loading.setStyleSheet(f"color: {TEXT_SUBTLE}; font-size: 14px; margin-top: 20px;")
        layout.addWidget(self.loading)

        self.dots = ["", ".", "..", "..."]
        self.i = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(450)

    def animate(self):
        self.loading.setText(f"Initializing{self.dots[self.i]}")
        self.i = (self.i + 1) % 4

# -----------------------------
# WORKER THREADS
# -----------------------------

class GenericRequestWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, target_func, *args, **kwargs):
        super().__init__()
        self.target_func = target_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.target_func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(e)

class FetchWorker(QThread):
    fetched = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, phone_number: str):
        super().__init__()
        self.phone_number = phone_number

    def run(self):
        try:
            url = f"{GET_CALLS_URL}/{self.phone_number}"
            print(f"Attempting to fetch calls from: {url}") # DEBUG PRINT
            
            # verify=False helps if the .exe cannot find SSL certs (Use with caution in production)
            # HTTP_TIMEOUT allows time for Render to wake up
            r = requests.get(url, timeout=HTTP_TIMEOUT) # Removed verify=False for security, but add it back if SSL error persists
            
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    self.fetched.emit(data)
                else:
                    print(f"Unexpected data format: {type(data)}")
                    self.fetched.emit([])
            else:
                print(f"Server returned status code: {r.status_code}")
                self.fetched.emit([])
        except requests.exceptions.ConnectionError:
            print("Connection Error: Could not reach Render server.")
            self.error.emit("Connection Error")
        except requests.exceptions.Timeout:
            print("Timeout: Render server took too long to respond (waking up?).")
            self.error.emit("Timeout")
        except Exception as e:
            print(f"Fetch Error: {e}")
            self.error.emit(str(e))

# -----------------------------
# TABLE MODEL
# -----------------------------
class CallsTableModel(QAbstractTableModel):
    def __init__(self, data=None):
        super().__init__()
        self.columns = ["selected", "name", "phone", "timestamp", "summary"]
        self.headers = ["", "Name", "Phone", "Timestamp", "Summary"]
        self._data = []
        if data:
            self.set_data(data)

    def set_data(self, data: List[Dict[str, str]]):
        self.beginResetModel()
        self._data = []
        for d in data:
            self._data.append({
                "selected": False,
                "name": d.get("client_name", "Unknown"),
                "phone": d.get("number", "Unknown"),
                "timestamp": d.get("timestamp", ""),
                "summary": d.get("summary", ""),
                "sid": d.get("sid")  # <--- CRITICAL: We must store the SID here
            })
        self.endResetModel()

    def data(self, index, role):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        key = self.columns[col]

        if role == Qt.ItemDataRole.CheckStateRole and key == "selected":
            return Qt.CheckState.Checked if self._data[row]["selected"] else Qt.CheckState.Unchecked

        if role == Qt.ItemDataRole.DisplayRole:
            if key == "selected":
                return ""
            return str(self._data[row][key])

        return None

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self.columns)

    def headerData(self, section, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            if 0 <= section < len(self.headers):
                return self.headers[section]
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == 0:
            return base | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable
        return base | Qt.ItemFlag.ItemIsEditable

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        row, col = index.row(), index.column()
        key = self.columns[col]

        if key == "selected" and (role == Qt.ItemDataRole.CheckStateRole or role == Qt.ItemDataRole.EditRole):
            checked = (value == Qt.CheckState.Checked) or (int(value) == 2) or (value is True)
            self._data[row]["selected"] = bool(checked)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
            return True
        return False

    def get_checked_rows(self):
        return [r for r in self._data if r["selected"]]

    def remove_checked(self):
        self.beginResetModel()
        self._data = [r for r in self._data if not r["selected"]]
        self.endResetModel()


# -----------------------------
# CALL PAGE
# -----------------------------
class CallAgentPage(QWidget):
    def __init__(self, user_id=None):
        super().__init__()
        self.phone_number = user_id 
        
        # Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # --- 1. PAGE TITLE ROW ---
        header_row = QHBoxLayout()
        
        title = QLabel("Call Agent")
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffffff;")
        header_row.addWidget(title)

        header_row.addStretch()

        # Status Badge
        self.info_label = QLabel("● Online")
        self.info_label.setStyleSheet("""
            color: #888; background-color: #151515; 
            border: 1px solid #333; border-radius: 12px; 
            padding: 4px 12px; font-size: 12px; font-weight: 600;
        """)
        header_row.addWidget(self.info_label)
        layout.addLayout(header_row)

        # --- 2. TABLE FRAME ---
        table_frame = QFrame()
        table_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {PANEL_BLACK};
                border: 1px solid #2a2a2a;
                border-radius: 8px;
            }}
        """)
        
        frame_layout = QVBoxLayout(table_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableView()
        self.model = CallsTableModel([])
        self.table.setModel(self.model)
        
        # --- HEADER & TABLE STYLING ---
        self.table.setStyleSheet(f"""
            QTableView {{
                background-color: transparent;
                border: none;
                gridline-color: #252525;
                selection-background-color: #2d4a60;
                selection-color: white;
            }}
            QTableView::item {{
                border-bottom: 1px solid #1e1e1e;
                color: #e0e0e0;
                height: 50px;
            }}
            /* --- THE HEADER BAR --- */
            QHeaderView {{
                background-color: #1a1a1a;
                border: none;
                border-bottom: 2px solid #333;
            }}
            QHeaderView::section {{
                background-color: #1a1a1a;
                color: #909090;
                font-weight: 600;
                font-size: 12px;
                text-transform: uppercase;
                padding: 5px;
                border: none;
                border-right: 1px solid #2a2a2a;
            }}
            QHeaderView::section:last {{
                border-right: none;
            }}
        """)
        
        # Table Behavior
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(55)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setShowGrid(False)

        # --- COLUMN CONFIGURATION ---
        header = self.table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Resize Modes
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(0, 45)
        
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch) # Name stretches
        
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(2, 140)
        
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(3, 100)
        
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch) # Summary stretches

        frame_layout.addWidget(self.table)
        layout.addWidget(table_frame)

        # --- 3. BOTTOM BUTTONS ---
        btn_row = QHBoxLayout()
        
        # SYNC BUTTON
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFixedWidth(100)
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.clicked.connect(self.refresh)
        
        # DELETE BUTTON (With Hover)
        self.del_btn = QPushButton("Delete")
        self.del_btn.setFixedWidth(100)
        self.del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.del_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a1a1a; 
                color: #ff6b6b; 
                border: 1px solid #5a2a2a;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #552222; /* Lighter Red Background */
                border: 1px solid #ff6b6b; /* Bright Red Border */
                color: #ffffff;            /* White Text */
            }
            QPushButton:pressed {
                background-color: #2a0a0a;
            }
        """)
        self.del_btn.clicked.connect(self.delete_checked)

        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.del_btn)
        btn_row.addStretch()
        
        layout.addLayout(btn_row)

        # Timers
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(AUTO_REFRESH_MS)
        QTimer.singleShot(300, self.refresh)

    def refresh(self):
        self.info_label.setText("● Refreshing")
        target_phone = self.phone_number
        if not target_phone: return

        worker = FetchWorker(target_phone)
        worker.fetched.connect(self.on_fetched)
        worker.error.connect(lambda: self.info_label.setText("● Offline"))
        worker.start()
        self.worker = worker

    def on_fetched(self, data):
        self.model.set_data(data)
        self.info_label.setText("● Online")

    def delete_checked(self):
        checked = self.model.get_checked_rows()
        if not checked: return
        if QMessageBox.question(self, "Delete", "Delete selected?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            sids = [r["sid"] for r in checked if r.get("sid")]
            threading.Thread(target=lambda: requests.post(DELETE_URL, json={"call_sids": sids}), daemon=True).start()
            QTimer.singleShot(500, self.refresh)


# -----------------------------
# ABOUT PAGE
# -----------------------------
class AboutPage(QWidget):
    def __init__(self, license_key="Unknown"):
        super().__init__()
        
        data = fetch_version_info() or {}
        displayed_version = data.get("version", CURRENT_VERSION)
        last_update = data.get("last_update", "—")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(25)

        title = QLabel("About")
        title.setStyleSheet("color: #e0e0e0; font-size: 24px; font-weight: bold;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignLeft)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #333; margin: 8px 0;")
        layout.addWidget(divider)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(10)

        def add_row(label_text, value_text):
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setStyleSheet("color: #a0a0a0; font-weight: 600; min-width: 140px;")
            value = QLabel(value_text)
            value.setStyleSheet("color: #e0e0e0; font-size: 14px;")
            row.addWidget(label)
            row.addWidget(value, 1)
            info_layout.addLayout(row)

        add_row("Application:", "VerityLink")
        add_row("Version:", displayed_version)
        add_row("License:", license_key) 
        add_row("Created By:", "VerityLink Communications")
        add_row("Last Update:", last_update)
        add_row("Description:", "AI-powered communication dashboard")

        layout.addLayout(info_layout)

        terms_btn = QPushButton("View Terms Conditions")
        terms_btn.setFlat(True)
        terms_btn.setStyleSheet("""
            QPushButton {
                color: #4da6ff;
                font-size: 13px;
                text-align: left;
                padding: 0;
                text-decoration: underline;
                background: transparent;
                border: none;
                outline: none;
            }
            QPushButton:hover {
                color: #79c0ff;
                background: transparent;
                border: none;
            }
            QPushButton:pressed {
                border: none;
            }
        """)
        terms_btn.clicked.connect(self.open_terms_dialog)
        layout.addWidget(terms_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        footer = QLabel("© 2025 VerityLink Communications. All rights reserved.")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color: #666; font-size: 11px; margin-top: 25px;")
        layout.addWidget(footer)

    def open_terms_dialog(self):
        dlg = TermsDialog(self)
        dlg.exec()

# -----------------------------
# CUSTOMIZE PAGE
# -----------------------------
class CustomizeAIPage(QWidget):
    def __init__(self, phone_number):
        super().__init__()
        self.phone_number = phone_number
        
        # Separate workers to avoid collision
        self.load_worker = None
        self.save_worker = None
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(40, 40, 40, 40)
        self.main_layout.setSpacing(20)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Agent Customization")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #e0e0e0;")
        self.main_layout.addWidget(title)

        display_number = self.phone_number if self.phone_number else "Unknown"
        subtitle = QLabel(f"Editing AI for Number: {display_number}")
        subtitle.setStyleSheet("font-size: 14px; color: #4da6ff; margin-bottom: 10px;")
        self.main_layout.addWidget(subtitle)

        form_container = QFrame()
        form_container.setStyleSheet(f"""
            QFrame {{
                background-color: {PANEL_BLACK};
                border-radius: 12px;
                border: none; 
            }}
        """)
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(0, 10, 0, 10)
        form_layout.setSpacing(25)

        def create_section(label_text, placeholder, height=100):
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #a0a0a0; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;")
            
            txt = QTextEdit()
            txt.setPlaceholderText(placeholder)
            txt.setFixedHeight(height)
            txt.setStyleSheet(f"""
                QTextEdit {{
                    background-color: #0f0f0f;
                    border: 1px solid #333;
                    border-radius: 8px;
                    color: #e0e0e0;
                    padding: 12px;
                    font-size: 13px;
                    line-height: 1.4;
                }}
                QTextEdit:focus {{
                    border: 1px solid #666;
                    background-color: #141414;
                }}
            """)
            return lbl, txt

        self.prompt_box_label, self.prompt_box = create_section(
            "System Personality", 
            "Define the agent's role, tone, and behavior constraints (e.g., 'You are a helpful receptionist named Lexi...')", 
            height=180
        )
        form_layout.addWidget(self.prompt_box_label)
        form_layout.addWidget(self.prompt_box)

        self.greeting_box_label, self.greeting_box = create_section(
            "Initial Greeting", 
            "What should the agent say immediately when the call connects? (e.g., 'Thanks for calling VerityLink, how can I help?')", 
            height=80
        )
        form_layout.addWidget(self.greeting_box_label)
        form_layout.addWidget(self.greeting_box)

        self.main_layout.addWidget(form_container)

        action_layout = QHBoxLayout()
        
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #2e8b57; font-weight: 500;")

        self.save_btn = QPushButton("Save Customization")
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setFixedWidth(200)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                color: #121212;
                font-weight: bold;
                border-radius: 8px;
                padding: 12px;
                font-size: 13px;
                border: none;
            }
            QPushButton:hover {
                background-color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #b0b0b0;
            }
        """)
        self.save_btn.clicked.connect(self.save)

        action_layout.addWidget(self.info_label)
        action_layout.addStretch()
        action_layout.addWidget(self.save_btn)

        self.main_layout.addLayout(action_layout)
        self.main_layout.addStretch()

        QTimer.singleShot(500, self.load_settings)

    def load_settings(self):
        if not self.phone_number:
            self.info_label.setText("Error: No phone number linked.")
            return
            
        # FIX: Check if loading already in progress
        if self.load_worker and self.load_worker.isRunning():
            return

        def fetch_logic():
            url = f"{SETTINGS_URL}/{self.phone_number}"
            r = requests.get(url, timeout=5)
            if r.ok:
                return r.json()
            return None

        self.load_worker = GenericRequestWorker(fetch_logic)
        self.load_worker.finished.connect(self.handle_load_result)
        self.load_worker.start()

    def handle_load_result(self, result):
        if isinstance(result, dict):
            self.prompt_box.setText(result.get("system_prompt", ""))
            self.greeting_box.setText(result.get("greeting", ""))
        elif isinstance(result, Exception):
            print(f"Error loading settings: {result}")
        else:
            pass

    def save(self):
        if not self.phone_number:
            QMessageBox.warning(self, "Error", "No phone number associated with this account.")
            return
            
        # FIX: Check if save already in progress
        if self.save_worker and self.save_worker.isRunning():
            return

        prompt = self.prompt_box.toPlainText().strip()
        greeting = self.greeting_box.toPlainText().strip()

        if not prompt or not greeting:
            QMessageBox.warning(self, "Missing Info", "Please fill in both fields.")
            return

        self.info_label.setText("Saving...")
        self.save_btn.setEnabled(False)

        def save_logic():
            payload = {
                "phone_number": self.phone_number,
                "system_prompt": prompt,
                "greeting": greeting
            }
            r = requests.post(SETTINGS_URL, json=payload, timeout=10)
            return r.status_code, r.ok

        self.save_worker = GenericRequestWorker(save_logic)
        self.save_worker.finished.connect(self.handle_save_result)
        self.save_worker.start()

    def handle_save_result(self, result):
        self.save_btn.setEnabled(True)
        if isinstance(result, tuple):
            status_code, ok = result
            if ok:
                self.info_label.setText("Settings Saved Successfully! ✅")
                QTimer.singleShot(3000, lambda: self.info_label.setText(""))
            else:
                self.info_label.setText(f"Save Failed: {status_code}")
        elif isinstance(result, Exception):
            self.info_label.setText(f"Connection Error: {result}")


# -----------------------------
# MAIN DASHBOARD WINDOW
# -----------------------------
class MainWindow(QWidget):
    SIDEBAR_EXPANDED_WIDTH = 220
    SIDEBAR_COLLAPSED_WIDTH = 50

    def __init__(self, user_data):
        super().__init__()
        self.user_data = user_data
        self.user_name = user_data.get("name", "User")
        self.user_id = user_data.get("id")
        self.phone_number = user_data.get("phone_number", "")
        self.license_key = user_data.get("license_key", "Unknown")
        
        # FIX: Initialize worker attributes BEFORE calling init_ui
        self.status_worker = None 
        self.toggle_worker = None
        
        self.setWindowTitle("VerityLink AI Dashboard")
        self.resize(1100, 700)
        self.init_ui()
        
        self.sidebar_expanded = True
        self.sidebar_anim = QPropertyAnimation(self.sidebar, b"minimumWidth")
        self.sidebar_anim.setDuration(400)
        self.sidebar_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.sidebar_anim.finished.connect(self._on_sidebar_anim_finished)
        
        # --- SYNC STATE ON STARTUP ---
        # This fixes the issue where it says "OFF" even if it should be "ON"
        QTimer.singleShot(1000, self.check_server_state)

    def init_ui(self):
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # --- Sidebar ---
        self.sidebar = QFrame()
        self.sidebar.setMinimumWidth(self.SIDEBAR_EXPANDED_WIDTH)
        self.sidebar.setMaximumWidth(self.SIDEBAR_EXPANDED_WIDTH)
        self.sidebar.setStyleSheet("background:#1e1e1e;")

        sb = QVBoxLayout(self.sidebar)
        sb.setContentsMargins(10, 10, 10, 10)
        sb.setSpacing(8)

        self.toggle_side = QToolButton()
        self.toggle_side.setText("☰")
        self.toggle_side.setStyleSheet("color:white; font-size:18px; background: transparent; border: none;")
        self.toggle_side.setFixedSize(36, 36)
        self.toggle_side.clicked.connect(self.toggle_sidebar)
        sb.addWidget(self.toggle_side, alignment=Qt.AlignmentFlag.AlignLeft)

        self.btn_call = QPushButton("Call Agent")
        self.btn_email = QPushButton("Email Agent")
        self.btn_draft = QPushButton("Customize AI")
        self.btn_about = QPushButton("About")

        for btn in [self.btn_call, self.btn_email, self.btn_draft, self.btn_about]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2b2b2b; 
                    color: white; 
                    text-align: center; 
                    padding: 10px; 
                    border: none;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #3c3c3c;
                }
                QPushButton:pressed {
                    background-color: #1a1a1a;
                }
            """)
            btn.setMinimumHeight(40)
            sb.addWidget(btn)
            
        self.btn_call.clicked.connect(lambda: self.stack.setCurrentWidget(self.call_page))
        self.btn_email.clicked.connect(lambda: QMessageBox.information(self, "Email Agent", "Email Agent page not implemented yet."))
        self.btn_draft.clicked.connect(lambda: self.stack.setCurrentWidget(self.customize_page))
        self.btn_about.clicked.connect(lambda: self.stack.setCurrentWidget(self.about_page))

        sb.addStretch()
        content_layout.addWidget(self.sidebar)

        # --- Pages ---
        self.stack = QStackedWidget()
        
        self.call_page = CallAgentPage(user_id=self.phone_number)
        self.call_page.phone_number = self.phone_number 
        
        self.about_page = AboutPage(license_key=self.license_key)
        self.customize_page = CustomizeAIPage(self.phone_number)
        
        self.stack.addWidget(self.call_page)
        self.stack.addWidget(self.about_page)
        self.stack.addWidget(self.customize_page)
        content_layout.addWidget(self.stack, 1)

        # --- Right Panel ---
        self.right_panel = QFrame()
        self.right_panel.setMinimumWidth(250)
        self.right_panel.setStyleSheet("background:#181818;")
        r = QVBoxLayout(self.right_panel)
        r.setContentsMargins(12, 12, 12, 12)
        r.addSpacing(12)

        self.toggle_btn = QPushButton("OFF")
        self.toggle_btn.setFixedHeight(44)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #b22222; 
                color: white; 
                font-weight: bold; 
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #d93333;
            }
        """)
        self.toggle_btn.clicked.connect(self.toggle_bot)
        r.addWidget(self.toggle_btn)

        r.addSpacing(20)
        
        tips_box = QFrame()
        tips_box.setObjectName("TipsCard")
        tips_box.setStyleSheet("""
            QFrame#TipsCard {
                background-color: #1f1f1f;
                border: 1px solid #2e2e2e;
                border-radius: 12px;
            }
            QLabel {
                background: transparent;
            }
            QLabel#TipsHeader {
                color: #e0e0e0;
                font-size: 15px;
                font-weight: 600;
            }
            QLabel#TipsBody {
                color: #d8d8d8;
                font-size: 13px;
                line-height: 1.4em;
            }
            QLabel#TipsFooter {
                color: #aaaaaa;
                font-size: 11px;
                margin-top: 8px;
            }
        """)

        tips_layout = QVBoxLayout(tips_box)
        tips_layout.setContentsMargins(14, 12, 14, 12)
        tips_layout.setSpacing(6)

        tips_header = QLabel("Quick Help", tips_box)
        tips_header.setObjectName("TipsHeader")
        tips_layout.addWidget(tips_header)

        tips_body = QLabel(
            "If dashboard appears offline:\n"
            "• Wait a few seconds, then refresh.\n"
            "• Check your internet connection.\n"
            "• Restart the dashboard if issue persists."
        )
        tips_body.setObjectName("TipsBody")
        tips_body.setWordWrap(True)
        tips_body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        tips_layout.addWidget(tips_body)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #333; margin: 6px 0;")
        tips_layout.addWidget(divider)

        tips_footer = QLabel("📞 Support: 403-775-7197  •\n      9 AM – 5 PM, 7 days/week")
        tips_footer.setObjectName("TipsFooter")
        tips_footer.setWordWrap(True)
        tips_layout.addWidget(tips_footer)

        r.addWidget(tips_box)
        r.addStretch()
        r.addWidget(QLabel("VerityLink, 2025", alignment=Qt.AlignmentFlag.AlignRight))
        content_layout.addWidget(self.right_panel)

        # --- Welcome banner ---
        top_banner = QFrame()
        top_banner.setStyleSheet("background:#111; border: none;")
        top_banner.setFixedHeight(60)

        banner_layout = QVBoxLayout(top_banner)
        banner_layout.setContentsMargins(0, 0, 0, 0)
        banner_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        welcome_label = QLabel(f"Welcome, {self.user_name}!", top_banner)
        welcome_label.setStyleSheet("color: #e0e0e0; font-size: 18px; font-weight: bold;")
        banner_layout.addWidget(welcome_label)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0,0,0,0)
        self.main_layout.setSpacing(0)
        self.main_layout.addWidget(top_banner)
        self.main_layout.addLayout(content_layout)

    def toggle_sidebar(self):
        if self.sidebar_anim.state() == self.sidebar_anim.State.Running:
            self.sidebar_anim.stop()

        if self.sidebar_expanded:
            self.sidebar_anim.setStartValue(self.SIDEBAR_EXPANDED_WIDTH)
            self.sidebar_anim.setEndValue(self.SIDEBAR_COLLAPSED_WIDTH)
            
            # During collapse, buttons remain visible and compress with the animation.
            
            self.sidebar_anim.start()
            self.sidebar_expanded = False
        else:
            # Show buttons before animation starts (for smooth expansion)
            for w in [self.btn_call, self.btn_email, self.btn_draft, self.btn_about]:
                w.show()
                
            self.sidebar_anim.setStartValue(self.SIDEBAR_COLLAPSED_WIDTH)
            self.sidebar_anim.setEndValue(self.SIDEBAR_EXPANDED_WIDTH)
            self.sidebar_anim.start()
            self.sidebar_expanded = True

    def _on_sidebar_anim_finished(self):
        if not self.sidebar_expanded:
            # Collapse finished. We must hide the buttons now to prevent interaction,
            # but this happens AFTER the animation is visually complete.
            for w in [self.btn_call, self.btn_email, self.btn_draft, self.btn_about]:
                 w.hide()
            
            self.sidebar.setMinimumWidth(self.SIDEBAR_COLLAPSED_WIDTH)
            self.sidebar.setMaximumWidth(self.SIDEBAR_COLLAPSED_WIDTH)
        else:
            # Expansion finished. Ensure buttons are shown (done in toggle_sidebar too, but safe here)
            for w in [self.btn_call, self.btn_email, self.btn_draft, self.btn_about]:
                w.show()
                
            self.sidebar.setMinimumWidth(self.SIDEBAR_EXPANDED_WIDTH)
            self.sidebar.setMaximumWidth(self.SIDEBAR_EXPANDED_WIDTH)

        self.call_page.updateGeometry()
        self.main_layout.update()
        
    def check_server_state(self):
        """Checks the actual status from the server on startup."""
        def task():
            try:
                url = f"{SETTINGS_URL}/{self.phone_number}"
                r = requests.get(url, timeout=10)
                if r.ok:
                    data = r.json()
                    is_active = data.get("active", False)
                    # This line MUST match the function name below
                    self.set_toggle_ui(is_active)
            except Exception as e:
                print(f"Failed to fetch initial state: {e}")
        
        threading.Thread(target=task, daemon=True).start()

    def toggle_bot(self):
        # Determine the target state (True = ON, False = OFF)
        target_state = self.toggle_btn.text() == "OFF"
        
        # Update UI immediately for responsiveness
        self.set_toggle_ui(target_state)
        
        # Send request in background with robust error handling
        threading.Thread(target=self._post_toggle, args=(target_state, self.phone_number), daemon=True).start()
        
        
        def toggle_task(active):
            requests.post(TOGGLE_URL, json={"active": active}, timeout=5)
            
        self.toggle_worker = GenericRequestWorker(toggle_task, target_state)
        self.toggle_worker.start()
        
    def _post_toggle(self, state, phone_number):
        """Sends the toggle command to the server with highly defensive exception handling."""
        
        # Use BaseException to catch lower-level errors that might crash the thread
        try:
            payload = {
                "active": state,
                "phone_number": phone_number
            }
            r = requests.post(TOGGLE_URL, json=payload, timeout=10)
            
            if not r.ok:
                # Log server-side issues
                print(f"Toggle failed: Server responded with HTTP status code {r.status_code}. Response: {r.text[:100]}")
                
        except requests.exceptions.RequestException as req_e:
            # Catch specific network/connection problems (DNS, timeout, connection refused)
            print(f"Toggle failed: Network Error - Could not reach server. Details: {req_e}")
            
        except BaseException as base_e:
            # Catch all other critical system or thread errors
            print(f"Toggle failed: CRITICAL THREAD ERROR - {base_e}")

    def set_toggle_ui(self, on):
        """Sets the visual state (color and text) of the ON/OFF button with hover effects."""
        
        if on:
            self.toggle_btn.setText("ON")
            # --- ON State Styling (Green) ---
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2e8b57; /* Sea Green */
                    color: white; 
                    font-weight: bold; 
                    border-radius: 8px; 
                    border: none;
                }
                QPushButton:hover {
                    background-color: #257045; /* Slightly darker green on hover */
                }
            """)
        else:
            self.toggle_btn.setText("OFF")
            # --- OFF State Styling (Red) ---
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #b22222; /* Dark Red */
                    color: white; 
                    font-weight: bold; 
                    border-radius: 8px; 
                    border: none;
                }
                QPushButton:hover {
                    background-color: #c23232; /* Slightly lighter red on hover */
                }
            """)

        def status_task():
            r = requests.get(STATUS_URL, timeout=3)
            if r.ok:
                return r.json()
            return None
            
        self.status_worker = GenericRequestWorker(status_task)
        self.status_worker.start()

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # --- SET APPLICATION ICON GLOBALLY ---
    icon_path = resource_path("logo6.png")
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)
    # -------------------------------------

    app.setStyleSheet(f"""
        QWidget {{
            background-color: {BASE_BLACK};
            color: {TEXT_MAIN};
            font-family: 'Segoe UI';
        }}

        QLabel {{
            color: {TEXT_MAIN};
        }}

        QPushButton {{
            background-color: #e0e0e0;
            color: #121212;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 13px;
            border: none;
        }}
        QPushButton:hover {{
            background-color: #ffffff;
        }}
        QPushButton:pressed {{
            background-color: #b0b0b0;
        }}

        QLineEdit, QTextEdit {{
            background-color: {PANEL_BLACK};
            color: {TEXT_MAIN};
            border: 1px solid {BORDER_GRAY};
            border-radius: 6px;
            padding: 6px;
        }}
        
        QTableView {{
            background-color: {PANEL_BLACK};
            alternate-background-color: #0f0f0f;
            color: {TEXT_MAIN};
            gridline-color: {BORDER_GRAY};
            selection-background-color: #303030;
            selection-color: white;
        }}
        
        QScrollBar:vertical {{
            border: none;
            background: {BASE_BLACK};
            width: 10px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: #333;
            min-height: 20px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: #444;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: none;
        }}
        QScrollBar:horizontal {{
            border: none;
            background: {BASE_BLACK};
            height: 10px;
            margin: 0px;
        }}
        QScrollBar::handle:horizontal {{
            background: #333;
            min-width: 20px;
            border-radius: 5px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: #444;
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: none;
        }}
    """)

    startup = StartupScreen()
    startup.resize(1100, 700)
    startup.show()

    def proceed_to_login():
        login_dialog = LoginDialog()
        if login_dialog.exec() == QDialog.DialogCode.Accepted:
            main_window = MainWindow(user_data=login_dialog.user_data)
            main_window.show()
            startup.close()
            app.main_window = main_window
        else:
            startup.close()
            sys.exit(0)

    QTimer.singleShot(1500, proceed_to_login)

    sys.exit(app.exec())

















