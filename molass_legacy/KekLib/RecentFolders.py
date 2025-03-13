# coding: utf-8
"""

    ファイル名：   RecentFolders.py

    処理内容：

       matplotlib 関連の追加クラス

"""
from datetime       import datetime

"""
    {
        'folder-1'  : [ '2016-03-10 08:44:04.286183', 'folder-1' ],
        'folder-2'  : [ '2016-03-10 08:44:04.395383', 'folder-2' ],
        'folder-3'  : [ '2016-03-10 08:44:04.504583', 'folder-3' ],
    }
"""

class RecentFolders:
    def __init__( self, size=None, settings=None ):
        if settings is None:
            import Settings
            settings = Settings

        if size is None:
            self.size       = settings.get_setting( 'num_recent_folders' )
        else:
            self.size       = size
        self.folder_dict    = settings.get_setting( 'recent_folders' )

    def add( self, path ):
        if len( self.folder_dict ) >= self.size:
            del self.folder_dict[ self.get_sorted_list()[-1][1] ]

        self.folder_dict[ path ] = [ str( datetime.now() ), path ]

    def get_sorted_list( self ):
        list_ = list( reversed( sorted( self.folder_dict.values() ) ) )
        # print( list_ )
        return list_

    def get_current_size( self ):
        return len( self.folder_dict )
