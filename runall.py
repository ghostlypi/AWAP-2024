import subprocess
import os

for f in os.listdir('maps'):
    print (f'Map {f}:')
    subprocess.run([f'python run_game.py -b bots/Player_v4.py -r bots/Player_v3.py -m maps/{f} --replay'], shell=True, cwd=os.getcwd())