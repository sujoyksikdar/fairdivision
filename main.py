# ATTENTION! Remember to set B in config.py correctly
# ATTENTION! Always check config.py and set the right parameters

import sys
import logging

from eq import *
from data import *


def main():
    eq_goods()
    return None


if __name__ == '__main__':
    FORMAT = "%(asctime)s %(levelname)s %(module)s %(lineno)d %(funcName)s:: %(message)s"
    logging.basicConfig(filename='common.log', filemode='a', level=logging.DEBUG, format=FORMAT)
    main()
