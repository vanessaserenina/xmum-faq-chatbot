import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
