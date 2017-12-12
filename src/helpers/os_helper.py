import os
from helpers.print_helper import *

def check_n_makedirs(path):
    if not os.path.exists(path):
        print_info("Creating folder: {}".format(path))
        os.makedirs(path)
