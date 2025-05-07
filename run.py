"""
Top‑level script to invoke preprocessing, training, and evaluation.
"""

import subprocess

if __name__ == '__main__':
    # 1. Preprocessing
    subprocess.run(['python', 'src/preprocess.py'])

    # 2. Train pipeline
    subprocess.run(['python', 'src/train.py'])

    # 3. Generate plots & report (in notebooks or scripts)
    subprocess.run(['jupyter', 'nbconvert', '--to', 'script', 'notebooks/analysis.ipynb'])
