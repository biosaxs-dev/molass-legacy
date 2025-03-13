# coding: utf-8
"""
    LmfitThreadSafe.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""
import threading
import lmfit
from lmfit                  import Parameters

lock = threading.Lock()

def minimize( *args, **kwargs ):
    with lock:
        result = lmfit.minimize( *args, **kwargs )
    return result

if __name__ == '__main__':
    import numpy    as np
    import time
    from nose.tools import eq_

    def func_threaded( result_list ):
        params = Parameters()
        params.add( 'A', value=0,       min=-1,  max=+1 )

        def obj_func( p ):
            A = p['A']
            array = np.array( [ A - 0.5 ] )
            time.sleep( 0.1 )
            return array

        result = minimize( obj_func, params, args=() )
        print( "result.params['A']=", result.params['A'].value )
        result_list.append( result.params['A'].value )

    thread_list = []
    result_list = []
    for i in range(3):
        thread = threading.Thread(
                    target=func_threaded,
                    name='Thread%d' % i,
                    args=[ result_list ],
                    )
        thread.start()
        thread_list.append( thread )

    for thread in thread_list:
        thread.join()

    print( 'result_list=', result_list )
    eq_( result_list, [ 0.5, 0.5, 0.5 ] )
