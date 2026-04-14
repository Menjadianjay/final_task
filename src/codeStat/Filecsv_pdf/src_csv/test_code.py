import sys
import os

# Get the current script's location and navigate to your root
current_file = os.path.abspath(__file__)
root = os.path.dirname(current_file)  # adjust with more os.path.dirname() as needed

sys.path.insert(0, root)