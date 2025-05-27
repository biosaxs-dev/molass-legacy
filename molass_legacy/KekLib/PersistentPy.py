"""
    PersistentPy.py - succcesor of PersistentInfo.py

    Copyright (c) 2024, Masatsuyo Takahashi, KEK-PF
"""
import os
import shutil
from molass_legacy.KekLib.BasicUtils import home_dir, exe_name
import molass_legacy.KekLib.CustomMessageBox as MessageBox

def kek_tools_folder():
    return '%s/.KekTools' % (home_dir())

def get_py_file_path(name):
    return '%s/%s/%s' % ( kek_tools_folder(), exe_name(), name )

def make_py_file_path(folder, file):
    if not os.path.exists( folder ):
        os.makedirs( folder )
    app_folder   = '%s/%s' % ( folder, exe_name() )
    if not os.path.exists( app_folder ):
        os.makedirs( app_folder )

    file_path = '%s/%s' % ( app_folder, file )
    if os.path.exists(file_path):
        # check if the file is writable
        assert os.access(file_path, os.W_OK)
    else:
        # check if the folder is writable
        try:
            access_test_file = os.path.join(app_folder, 'access_test')
            fh = open(access_test_file, 'w')
            fh.close()
            os.remove(access_test_file)
        except Exception as exc:
            print('access error in', app_folder, exc)
            raise

    return file_path

def load_py_file(filename, locals_={}):
    from molass_legacy.KekLib.EvalUtils import eval_file
    return eval_file(filename, locals_)

class PersistentPy:
    def __init__(self, filename, defaults={}, py_file=None, alt_folder=None, locals_=None, warn_on_fail=True):
        self.alt_folder = alt_folder
        self._dict = defaults
        if py_file is None:
            try:
                if alt_folder is None:
                    folder = kek_tools_folder()
                else:
                    folder = alt_folder

                py_file = make_py_file_path(folder, filename)
                self.alt_folder = alt_folder
            except:
                try:
                    alt_folder = __file__
                    for k in range(3):
                        alt_folder, _ = os.path.split(alt_folder)
                        print([k], alt_folder)
                    alt_folder = os.path.join(alt_folder, '.KekTools')
                    py_file = make_py_file_path(alt_folder, filename)
                    MessageBox.showwarning( 'Alternative Settings Folder Notification',
                        'Using %s\n'
                        'as your settings folder '
                        'due to an unidentified error.\n'
                        % alt_folder
                        )
                    self.alt_folder = alt_folder
                except:
                    MessageBox.showerror( 'No other alternative',
                        'Failed to use %s\n' % alt_folder
                        )
                    assert False

        self.py_file = py_file

        if os.path.exists(self.py_file) and os.path.getsize(self.py_file):
            try:
                self._dict = load_py_file(self.py_file, locals_=locals_)
            except Exception as exc:
                if True:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning("in pid=%d, load_py_file(%s) failed due to %s", os.getpid(), self.py_file, exc)
                else:
                    from molass_legacy.KekLib.ExceptionTracebacker import log_exception
                    etb = log_exception(None, "in pid=%d, load_py_file(%s): " % (os.getpid(), self.py_file), n=10)

                if warn_on_fail:
                    MessageBox.showwarning( 'Broken Settings Notification',
                        'Your settings infomation previously stored in\n'
                        '"%s" seems to have been broken.\n'
                        '\n'
                        'Sorry, your entire settings information will be lost\n'
                        'and reset to the default.\n'
                        '\n'
                        'This inconvenience can happen in an invocation just after such cases as listed.\n'
                        '  - abnormal termination of the program\n'
                        '  - multiple simutaneous invocation of the program\n'
                        % (self.py_file) )
                            # TODO: set parent

                py_dir = os.path.dirname(self.py_file)
                try:
                    # shutil.rmtree(py_dir)
                    pass
                except:
                    # continue anyway
                    pass
                self._dict = {}
        else:
            self._dict = {}

        for k in defaults.keys():
            if self._dict.get(k) == None:
                self._dict[k] = defaults[k]

    def set_dictionary(self, dictionary):
        self._dict = dictionary

    def get_dictionary(self):
        return self._dict

    def save(self, file=None, pprint=True):
        if file is None:
            file = self.py_file
        with open(file, "w") as fh:
            if pprint:
                import pprint
                fh.write(pprint.pformat(self._dict))
            else:
                fh.write(str(self._dict))