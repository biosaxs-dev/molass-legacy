"""
    run.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""

if __name__ == '__main__':
    import os
    import sys
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append( this_dir + '/..' )
    from Processes.Main import main
    main()
