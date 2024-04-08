import os
from pathlib import Path
import sys


def append_gym_checker():
    pwd = Path(os.getcwd())
    if pwd.name == "notebooks":
        sys.path.append(str(pwd.parent / "gym-checkers"))
    else:
        sys.path.append(str(pwd / "gym-checkers"))
