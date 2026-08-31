"""
    optimizer.py
"""
# Headless subprocess -- never displays anything, but without forcing Agg,
# matplotlib auto-resolves to TkAgg (Windows default) since Tkinter is always
# importable. Any plotting inside the legacy pipeline that isn't strictly on
# the main thread then risks Tcl_AsyncDelete. Must run before any other
# import that could trigger matplotlib's lazy backend resolution.
import matplotlib
matplotlib.use('Agg')

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
