"""
Vercel-compatible handler for the Flask app
Place this file at the root level for Vercel deployment
"""

import sys
import os

# Add the scripts directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app import app

# Vercel expects a handler function or app variable
def handler(environ, start_response):
    return app(environ, start_response)

# Alternative: export the app directly
application = app

if __name__ == "__main__":
    app.run()
