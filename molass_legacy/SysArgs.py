"""
Global variable to hold command line arguments.
"""
from argparse import ArgumentParser

sys_args = None

def parse_sys_args():
    """
    Parse command line arguments and set global sys_args variable.
    """
    global sys_args

    ap = ArgumentParser()
    ap.add_argument('--devel', action='store_true', help='Enable development mode')
    ap.add_argument('--debug', action='store_true', help='Enable debug mode')
    sys_args = ap.parse_args()