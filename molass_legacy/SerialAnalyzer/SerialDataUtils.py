"""
    SerialDataUtils.py

    Copyright (c) 2018-2025, SAXS Team, KEK-PF
"""
import os
import re
import glob
import numpy as np
from molass_legacy.KekLib.BasicUtils import clear_dirs_with_retry
from molass_legacy.KekLib.NumpyUtils import np_loadtxt, np_savetxt
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting, set_path_length
from molass_legacy.SerialAnalyzer.DevSettings import get_dev_setting
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker, log_exception

file_name_filter_re = re.compile( r'(UV|spectra)[\w-]*\.txt$', re.IGNORECASE )
file_name_re    = re.compile( r'\D(\d+).txt$' )
data_line_re    = re.compile( r'^\d' )

warned_folders = {}

def get_xray_files( in_folder, file_extension=None, name_check=True ):
    if in_folder is None:
        return []

    if file_extension is None:
        file_extension = get_setting( 'file_extension' )

    datafiles = sorted( glob.glob( in_folder + '/' + file_extension ) )
    if not name_check:
        return datafiles

    f_ext = file_extension.replace( '*', '\\' )
    xray_file_re    = re.compile( r'_(\d{5})\w*' + f_ext + '$' )

    ret_files = []
    err_files = []
    for k, f in enumerate( datafiles ):
        m = xray_file_re.search( f )
        if m:
            n = int(m.group(1))
            # assert n == k
            ret_files.append( f )
        else:
            err_files.append( f )

    if len(err_files) < len(ret_files):

        if len(err_files) > 0:
            already_warned = warned_folders.get(in_folder)
            suppress_warning = get_setting( 'suppress_warning' )
            test_pattern = get_setting("test_pattern")
            if already_warned is None and not suppress_warning and test_pattern is None:
                import molass_legacy.KekLib.OurMessageBox as MessageBox
                files = [ os.path.split( f )[1] for  f in err_files ]
                MessageBox.showwarning( "Warning", "irregular filenames " + str(files)
                                        + " in " + in_folder + " have been ignorged." )
                warned_folders[in_folder] = 1

        return ret_files
    else:
        return datafiles

def get_file_no( name ):
    m = file_name_re.search( name )
    if m:
        return int( m.group(1) )
    else:
        return None

def find_conc_files( in_folder ):
    files = filter( lambda x: file_name_filter_re.search( x ), glob.glob( in_folder + '/*.txt' ) )
    file_recs = map( lambda x: [ get_file_no( x ), x ], files )
    datafile_recs = sorted( file_recs )
    return datafile_recs

def get_uv_filename( uv_folder, glob=None ):
    disable_uv_data = get_setting('disable_uv_data')
    if disable_uv_data:
        return None

    datafile_recs  = list( find_conc_files( uv_folder ) )
    # print( 'datafile_recs=', datafile_recs )
    if len(datafile_recs) > 0:
        dir_, file = os.path.split( datafile_recs[0][1] )
        if len(datafile_recs) == 1:
            return file
        else:
            return '*.txt'
    else:
        return try_to_find_uv_in_any_txt(uv_folder)

def try_to_find_uv_in_any_txt(uv_folder):
    ret_file = None
    for file in glob.glob(uv_folder + '/*.txt'):
        try:
            found =False
            value_count = 0
            fh = open(file, 'r', encoding='cp932')
            for k, line in enumerate(fh):
                # print([k], line)
                m = data_line_re.match(line)
                if m:
                    vec = np.array([float(v) for v in line.split("\t")[:-1]])
                    if vec[0] > 190 and len(vec) > 100:
                        value_count += 1
                        if value_count > 10:
                            found = True
                            folder, ret_file = os.path.split(file)
                            break
            fh.close()
        except:
            log_exception(None, "try_to_find_uv_in_any_txt: ")
            continue
    return ret_file

def get_mtd_filename(in_folder):
    for folder in [in_folder, os.path.abspath(in_folder + r'\..')]:
        files = glob.glob(folder + r'\*.mtd')
        if len(files) > 0:
            break

    if len(files) == 0:
        name = None
    else:
        name = files[0]
    return name

def load_uv_file(file, column_header=False, return_dict=False):
    fh = open( file, 'r', encoding='cp932' )

    comment_lines = []

    def generator( fh ):
        for line in fh:
            if data_line_re.match( line ):
                yield line
            else:
                comment_lines.append( line )

    data = np.loadtxt( generator( fh ) )
    fh.close()

    # print( 'comment_lines=', comment_lines )

    uv_date_line_re = re.compile( r'^Date\s+(\d+)年(\d+)月(\d+)日' )
    uv_device_line_re = re.compile( r'^Spectrometers:\s+(\w+)' )

    measurement_date = None
    uv_device_no = None

    for line in comment_lines:
        m = uv_date_line_re.match(line)
        if m:
            measurement_date = int(m.group(1) + m.group(2) + m.group(3))
            # print('uv measurement_date=', measurement_date)
            continue
        m = uv_device_line_re.match(line)
        if m:
            uv_device_no = m.group(1)
            break

    set_setting("uv_device_no", uv_device_no)
    set_path_length(measurement_date, uv_device_no)

    if return_dict:
        return dict(data=data,
                    comment_lines=comment_lines,
                    measurement_date=measurement_date,
                    uv_device_no=uv_device_no)

    if column_header and len( comment_lines ) > 1:
        header = comment_lines[-2]
        col_header = header.split( '\t' )[0:-1]     # changed from [1:-1]
        # print( 'len(col_header)=', len(col_header) )
        # print( 'data.shape=', data.shape )
        return data, col_header
    else:
        return data

def load_intensity_files( in_folder, logger=None, return_comments=False, debug=False ):
    datafiles = get_xray_files( in_folder )
    data_array, comments = load_xray_files( datafiles, debug=debug )
    if return_comments:
        return data_array, datafiles, comments
    else:
        return data_array, datafiles

def load_xray_files(datafiles_, return_lacking_info=False, qv=None, uniform_qv=True, logger=None, debug=False ):
    data_list = []
    comments = None
    encoding = 'cp932'
    size_list = []
    for k, file in enumerate(datafiles_):
        try:
            data, _ = serial_np_loadtxt( file, encoding=encoding )
            if debug:
                print([k], file, data.shape)

            size_list.append(data.shape[0])

            if comments is None:
                comments = _

        except UnicodeDecodeError as exc:
            etb = ExceptionTracebacker()
            encoding = 'utf-8'
            retry_message = "retrying with encoding = '%s'" % encoding
            if logger is None:
                print(etb)
                print(retry_message)
            else:
                logger.warning(retry_message)
            data, _ = serial_np_loadtxt( file, encoding=encoding )
        except Exception as exc:
            print( "Error: Can't read " + file + '\n\t' + str(exc))
            # TODO: exe log gui
            etb = ExceptionTracebacker()
            print(etb)
            continue
        data_list.append( data )

    if uniform_qv:
        size_array = np.array(size_list)
        minn = np.min(size_array)
        maxn = np.max(size_array)
        if minn < maxn or qv is not None:
            data_list, lacking_list, minn_qv = convert_to_the_least_shape(data_list, size_array, minn, maxn, qv=qv)
        else:
            lacking_list = []
            minn_qv = None
    else:
        pass

    measurement_date = None
    sangler_version = None
    if comments is not None:
        for line in comments:
            if line.find('Date') > 0:
                date_re = re.compile(r'(\d+/\d+/\d+)')
                m = date_re.search(line)
                if m:
                    measurement_date = int(m.group(1).replace('/', ''))
            elif line.find('SAngler Version') > 0:
                version_re = re.compile(r':\s(\S+)')
                m = version_re.search(line)
                if m:
                    sangler_version = m.group(1)
                    break

    set_setting('measurement_date', measurement_date)
    set_setting('sangler_version', sangler_version)
    if return_lacking_info:
        return np.array( data_list ), comments, (minn, maxn, lacking_list, minn_qv)
    else:
        return np.array( data_list ), comments

def convert_to_the_least_shape(data_list, size_array, minn, maxn, qv=None):
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Converting the input data set by adjusting to the intersecting q-values with (minn, maxn)=(%d, %d)", minn, maxn)

    wh = np.where(size_array == maxn)[0]
    maxn_qv = data_list[wh[0]][:,0]

    if qv is None:
        wh = np.where(size_array == minn)[0]
        minn_qv = data_list[wh[0]][:,0]
    else:
        minn_qv = qv

    while True:
        ok = True
        ret_list = []
        lacking_list = []

        for k, data in enumerate(data_list):
            isect_qv, _, isect_ind = np.intersect1d(minn_qv, data[:,0], return_indices=True)
            if data.shape[0] < maxn:
                # 
                d = np.setdiff1d(np.union1d(maxn_qv, data[:,0]), np.intersect1d(maxn_qv, data[:,0]))
                lacking_list.append((k, data.shape[0], list(d)))
            if np.array_equal(isect_qv, minn_qv):
                ret_list.append(np.array([minn_qv, data[isect_ind,1], data[isect_ind,2]]).T)
            else:
                logger.warning("retry at %d-th file.", k)
                ok = False
                minn_qv = isect_qv
                break

        if ok:
            break

    logger.warning("Files with lacking q-values were, in (file no, size, lacking Q's) list,")
    for pair in lacking_list:
        logger.warning("\t%s", str(pair))

    logger.warning("The number of q-values has been changed from %d to %d finally.", maxn, len(minn_qv))
    set_setting('found_lacking_q_values', 1)

    return ret_list, lacking_list, minn_qv

def load_uv_array( conc_folder, conc_file=None, column_header=False ):
    disable_uv_data = get_setting('disable_uv_data')

    if disable_uv_data:
        datafile_recs = []
        conc_file = None
    else:
        if conc_file is None:
            if not os.path.exists( conc_folder ):
                raise Exception( conc_folder + ' does not exist!' )

            datafile_recs = find_conc_files( conc_folder )
            # print( 'datafile_recs=', datafile_recs )
            conc_file = conc_folder + '/*.txt'
        else:
            if conc_folder is None:
                conc_file_path = conc_file
            else:
                conc_file_path = os.path.join( conc_folder, conc_file  )
            if not os.path.exists( conc_file_path ):
                raise Exception( conc_file_path + ' does not exist!' )
            datafile_recs = [ [ 1, conc_file_path ] ]
            conc_file = conc_file_path

    data_array = None
    col_header = None
    lvector = None

    for no, file in datafile_recs:
        if col_header is None:
            data, col_header = load_uv_file( file, column_header=True )
        else:
            data = load_uv_file( file )

        if data_array is None:
            lvector     = data[ :, 0 ] 
            data_array  = data[ :, 1: ] 
        else:
            data_array = np.hstack( ( data_array, data[ :, 1: ] ) )

    if column_header:
        return data_array, lvector, conc_file, col_header
    else:
        return data_array, lvector, conc_file

def sa_uv_loadtxt(path):
    folder, file = os.path.split(path)
    ret = load_uv_array(folder, file, column_header=True)
    return ret[0], ret[1], ret[3]

def sa_uv_savetxt(path, vector, data, header):
    with open(path, "w", newline="\n") as fh:
        fh.write("編集データ\n")
        fh.write(">>>>>>>>>>>>>> 連続計測（領域波長）Data Start<<<<<<<<<<<<\n")
    np_savetxt(path, np.vstack([vector, data.T]).T, mode="a", column_names=header)
    with open(path, "a", newline="\n") as fh:
        fh.write(">>>>>>>>>>>>>> Data End <<<<<<<<<<<<\n")

def serial_np_loadtxt( filename, encoding='cp932' ):
    if filename.find( '.int' ) > 0:
        return np_loadtxt_intensity( filename, encoding=encoding )
    else:
        return np_loadtxt( filename, encoding=encoding )

def np_loadtxt_intensity( filename, encoding='cp932' ):
    fh = open( filename, encoding=encoding )

    comment_lines = []

    def generator( fh ):
        for line in fh:
            if len( comment_lines ) < 3:
                comment_lines.append( line )
            else:
                yield line

    try:
        ret = np.loadtxt( generator( fh ) )
        array = ret[ :, 1:4 ]
    except:
        etb = ExceptionTracebacker()
        etb.log()
        array = None

    fh.close()
    return array, comment_lines

def save_xray_base_profiles( serial_data, corrected_base ):
    print( 'save_xray_base_profiles: corrected_base.shape=', corrected_base.shape )

    qvector = serial_data.qvector
    qsize   = len(qvector)
    qvector_ = qvector.reshape( (1, qsize) )
    x_array = serial_data.intensity_array

    xb_folder           = get_dev_setting( 'xb_folder' )
    base_file_postfix   = get_dev_setting( 'base_file_postfix' )
    postfix_ext         = base_file_postfix + '.dat'

    clear_dirs_with_retry( [xb_folder] )

    for i, file in enumerate( serial_data.datafiles ):
        # print( file )
        d, f = os.path.split( file )
        f_  = f.replace( '.dat', postfix_ext )
        out_file = os.path.join( xb_folder, f_ ).replace( '/', '\\' )
        base_data   = np.vstack( [ qvector_, corrected_base[:,i].reshape( (1, qsize) ), x_array[i,:,2] ] ).T
        np_savetxt( out_file, base_data )
