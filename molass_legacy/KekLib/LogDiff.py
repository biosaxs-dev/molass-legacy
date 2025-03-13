"""

    ファイル名：    LogDiff.py

    処理内容：      ログファイルの diff

    Copyright (c) 2017,2023, Masatsuyo Takahashi, KEK-PF

"""
import re
from difflib    import unified_diff
from io         import StringIO

log_diff_re = re.compile( r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' )
time_re     = re.compile( r'was \d\.\d+ seconds' )

def time_re_sub( line ):
    return re.sub( time_re, 'was S.SSS seconds', line )

def make_log_diff(from_file, to_file, re_sub=None, write_substituted=False):

    def substitute( line ):
        line = re.sub( log_diff_re, 'YYYY-mm-dd HH:MM:SS', line )
        if re_sub is not None:
            line = re_sub( line )
        return line

    log_text_list = []
    for file in [from_file, to_file]:
        fh = open( file )
        text = [ substitute(line) for line in fh ]
        if write_substituted:
            path = file.replace(".log", ".txt")
            with open(path, "w") as out_fh:
                out_fh.write("".join(text))
        log_text_list.append( text )

    buff = StringIO()
    buff.writelines( unified_diff( log_text_list[0], log_text_list[1], fromfile=from_file, tofile=to_file ) )

    return buff.getvalue()

if __name__ == '__main__':
    import sys
    print(sys.argv)
    assert len(sys.argv) == 3
    print(make_log_diff(*sys.argv[1:3], write_substituted=True))
