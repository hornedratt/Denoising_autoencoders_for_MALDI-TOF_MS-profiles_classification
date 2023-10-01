import os
import pandas as pd

# Getting the current file directory
current_dir = os.path.dirname(__file__)

# Constructing the relative path to the target file
relative_path = os.path.join(current_dir, '../../data/raw/profile.csv')

# Resolving the relative path to absolute
absolute_path = os.path.abspath(relative_path)

# Checking if the file exists before reading it
if os.path.exists(absolute_path):
    df = pd.read_csv(absolute_path)
    print(df)
else:
    print(f"File not found: {absolute_path}")