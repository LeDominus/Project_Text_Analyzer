import sys
import os
import webbrowser
from threading import Timer

def add_module_path():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")

    app_path = os.path.join(base_path, 'app')
    if app_path not in sys.path:
        sys.path.insert(0, app_path)

add_module_path()

from app import app

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run()


