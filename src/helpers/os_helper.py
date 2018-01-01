import os
import shutil
from helpers.print_helper import *
from tqdm import tqdm

def check_n_makedirs(path):
    if not os.path.exists(path):
        print_info("Creating folder: {}".format(path))
        os.makedirs(path)

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in tqdm(os.listdir(src)):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

