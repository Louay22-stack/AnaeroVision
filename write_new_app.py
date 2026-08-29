import urllib.request
import os

# Just verify we're in the right place
print("Current directory:", os.getcwd())
print("Files here:", [f for f in os.listdir() if f.endswith('.py')])