"""
    optimizer.py
"""

def main():
    import os
    import sys
    python_syspath = os.environ.get('MOLASS_PYTHONPATH')
    if python_syspath is None:
        this_dir = os.path.dirname( os.path.abspath( __file__ ) )
        root_dir = os.path.dirname(os.path.dirname( this_dir ))
        sys.path.insert(0, root_dir)
    else:
        for path in python_syspath.split(os.pathsep):
            if path not in sys.path:
                sys.path.insert(0, path)
    from molass_legacy.Optimizer.OptimizerMain import main_driver
    main_driver()

if __name__ == '__main__':
    main()
