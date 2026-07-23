import os
import subprocess

data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

subprocess.run([
    'kaggle', 'datasets', 'download', '-d',
    'arenagrenade/the-complete-pokemon-images-data-set', 
    '-p', data_dir, '--unzip'
])