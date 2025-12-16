from streamlit.web import cli as stcli
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "app.py"

    # Mimic: streamlit run path/to/app.py
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
    ]

    stcli.main()
