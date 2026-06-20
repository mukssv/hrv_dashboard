import os
import subprocess
import webbrowser
import time

project_dir = os.path.dirname(os.path.abspath(__file__))

python_exe = os.path.join(
    project_dir,
    "venv",
    "Scripts",
    "python.exe"
)

subprocess.Popen(
    [
        python_exe,
        "-m",
        "streamlit",
        "run",
        "app.py"
    ],
    cwd=project_dir
)

time.sleep(5)

webbrowser.open("http://localhost:8501")