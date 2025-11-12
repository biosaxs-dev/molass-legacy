"""
    optimizer-dummy.py
"""

def main():
    import os
    import sys
    import getopt
    this_dir = os.path.dirname( os.path.abspath( __file__ ) )
    root_dir = os.path.dirname(os.path.dirname( this_dir ))
    sys.path.insert(0, root_dir)

    optlist, args = getopt.getopt(sys.argv[1:], 'c:w:f:n:i:b:d:m:s:r:t:p:T:M:S:')
    print(optlist, args)
    optdict = dict(optlist)

    from molass_legacy.KekLib.ChangeableLogger import Logger
    from molass_legacy._MOLASS.Version import get_version_string

    work_folder = optdict['-w']
    os.chdir(work_folder)
    work_folder = os.getcwd()   # to get absolute path

    nnn = int(work_folder[-3:])

    log_file = "optimizer.log"
    logger = Logger(log_file)

    logger.info(get_version_string(with_date=True))

    with open("pid.txt", "w") as fh:
        fh.write("pid=%d\n" % os.getpid())

    import time
    time.sleep(60)  # Dummy wait to simulate optimization process

if __name__ == '__main__':
    main()