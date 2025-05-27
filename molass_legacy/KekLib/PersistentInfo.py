"""

    PersistentInfo.py

    Copyright (c) 2016-2023, Masatsuyo Takahashi, KEK-PF

"""
import sys
import os
import pickle
import shutil
if sys.version_info > (3,):
    read_open_mode  = 'rb'
    write_open_mode = 'wb'
else:
    read_open_mode  = 'r'
    write_open_mode = 'w'

from molass_legacy.KekLib.BasicUtils     import home_dir, exe_name

def kek_tools_folder():
    return '%s/.KekTools' % ( home_dir() )

def get_pickle_file_path( name ):
    return '%s/%s/%s' % ( kek_tools_folder(), exe_name(), name )

def make_pickle_file_path( folder, file ):
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

class PersistentInfo:
    def __init__( self, filename, defaults={}, pickle_file=None, alt_folder=None ):
        self.alt_folder = None

        if pickle_file is None:
            try:
                if alt_folder is None:
                    folder = kek_tools_folder()
                else:
                    folder = alt_folder

                pickle_file = make_pickle_file_path(folder, filename )
                self.alt_folder = alt_folder
            except:
                try:
                    import molass_legacy.KekLib.CustomMessageBox as MessageBox
                except:
                    import OurMessageBox as MessageBox
                try:
                    alt_folder = __file__
                    for k in range(3):
                        alt_folder, _ = os.path.split(alt_folder)
                        print([k], alt_folder)
                    alt_folder = os.path.join(alt_folder, '.KekTools')
                    pickle_file = make_pickle_file_path(alt_folder, filename )
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

        self.pickle_file = pickle_file

        # パラメータのロード
        if os.path.exists( self.pickle_file ) and os.path.getsize( self.pickle_file ):
            try:
                pf = open( self.pickle_file, read_open_mode )
                self.dictionary = pickle.load( pf )
                pf.close()
            except:
                from molass_legacy.KekLib.ExceptionTracebacker import log_exception
                etb = log_exception(None, "open( self.pickle_file): ", n=10)

                pf.close()
                pickle_dir = os.path.dirname( self.pickle_file )

                if etb.last_lines().find("AttributeError: Can't get attribute") >= 0:
                    # temporary fix to avoid inconvenience for users with a message like the following
                    # AttributeError: Can't get attribute 'FlowChange' on <module 'Trimming.FlowChange' from '...\\FlowChange.py'>
                    # consider to remove this cause
                    pass
                else:
                    try:
                        import molass_legacy.KekLib.CustomMessageBox as MessageBox
                    except:
                        import OurMessageBox as MessageBox
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
                        % ( self.pickle_file ) )
                        # TODO: set parent
                try:
                    shutil.rmtree( pickle_dir )
                except:
                    # continue anyway
                    pass
                self.dictionary = {}
        else:
            self.dictionary = {}

        for k in defaults.keys():
            if self.dictionary.get(k) == None:
                self.dictionary[k] = defaults[k]

    def set_dictionary( self, dictionary ):
        self.dictionary = dictionary

    def get_dictionary( self ):
        return self.dictionary

    def save( self, file=None ):
        if file is None:
            file = self.pickle_file
        pf = open( file, write_open_mode )
        pickle.dump( self.dictionary, pf )
        pf.close()
