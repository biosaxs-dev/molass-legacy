"""

    ProgressInfo.py

    Copyright (c) 2017-2023, Masatsuyo Takahashi, KEK-PF

"""
import queue
import time

INFO_TYPE_STEP  = 0
INFO_TYPE_MAX   = 1

STATE_INIT      = 0
STATE_DOING     = 1
STATE_DONE      = 2
SAFE_FINISH     = 3
STATE_ERROR     = -1
STATE_FATAL     = -2

exe_queue  = queue.Queue()
cmd_queue  = queue.Queue()

def cleanup_queues():
    for q in [ exe_queue, cmd_queue ]:
        while not q.empty():
            q.get()

"""
    stream      phase       max
    Guinier     0   autorg  num_files
                1   excel   openpyxl(), com()

    Zero-Ex     0   asc(0)  
                1   desc(0) 
                2   asc(1)  
                3   desc(1) 
                .
                .   merge   com()
"""

class StreamState:
    def __init__( self, phase, step, state ):
        self.phase  = phase
        self.step   = step
        self.state  = state

class ExecInfo:
    def __init__( self, **entries ): 
        self.__dict__.update(entries)

        """
        self.info_type  :   0 step change
                            1 max change

        self.stream     :   0 
                            1 
        self.phase      :   0, 1, 2
        self.step       :   0, 1, 2, ..., num_steps
        self.state      :   
        self.message    :   
        """

    def __str__( self ):
        return str( self.state )

class ProgressInfo:
    def __init__( self, init_max_dict ):
        self.stream_max_dict        = {}
        self.stream_cur_dict        = {}
        self.stream_end_phase_dict  = {}
        self.g_cumulative_max_dict  = {}
        self.stream_phases          = {}
        self.stream_time_dict       = {}
        self.g_max_dict             = init_max_dict
        for k, v in sorted( self.g_max_dict.items() ):
            stream  = k[0]
            phase   = k[1]
            smax = self.stream_max_dict.get( stream )
            if smax is None:
                smax = v
            else:
                smax += v
            self.g_cumulative_max_dict[k]       = smax
            self.stream_max_dict[stream]        = smax
            self.stream_end_phase_dict[stream]  = phase

        self.stream_state_dict = {}
        for stream in self.stream_max_dict.keys():
            self.stream_state_dict[stream]  = STATE_INIT
            self.stream_cur_dict[stream]    = 0
            self.stream_phases[stream]      = {}

        self.current_stream = 0

    def get_start_time( self, stream ):
        rec = self.stream_time_dict.get(stream)
        return None if rec is None else rec[0]

    def get_info( self, block=False ):
        """
            note that get_info is called only in the main thread
            so that there is no need to care for thread safety.
        """

        try:
            info = exe_queue.get( block )
        except queue.Empty:
            info = None

        if info is None: return info

        if info.info_type == INFO_TYPE_MAX:
            new_progress_info = ProgressInfo( info.max_dict )
            # assumed this info is given prior to info.stream step info
            # and info.max_dict includes info.stream's max info only
            self.stream_max_dict[info.stream]       = new_progress_info.stream_max_dict[info.stream]
            self.stream_end_phase_dict[info.stream] = new_progress_info.stream_end_phase_dict[info.stream]
            assert( self.stream_state_dict[info.stream] == STATE_INIT )
            assert( self.stream_cur_dict[info.stream] == 0 )
            for k, v in info.max_dict.items():
                self.g_max_dict[k]              = v
                self.g_cumulative_max_dict[k]   = new_progress_info.g_cumulative_max_dict[k]
            print( 'max info changed by', info.max_dict )
            return

        if info.phase >= 0:
            self.stream_phases[info.stream][info.phase] = 1
            time_rec = self.stream_time_dict.get(info.stream)
            if time_rec is None:
                self.stream_time_dict[info.stream] = [ time.time(), None ]

        if info.state is None:
            end_phase = self.stream_end_phase_dict[info.stream]
            if info.phase >= end_phase:
                num_steps = self.g_max_dict.get( (info.stream, info.phase) )
                if num_steps is None:
                    # TODO
                    assert( False )
                if info.step >= num_steps:
                    state = STATE_DONE
                    time_rec = self.stream_time_dict.get(info.stream)
                    time_rec[1] = time.time()
                else:
                    state = STATE_DOING
            else:
                state = STATE_DOING
        else:
            # info.state is treated as stronger info than info.step
            state = info.state

        if info.phase == 0:
            cumulative_num_steps    = 0
        else:
            try:
                last_phase = sorted( self.stream_phases[info.stream].keys() )[-2]
                cumulative_num_steps    = self.g_cumulative_max_dict[ (info.stream, last_phase ) ]
            except IndexError as exc:
                # print( self.stream_phases )
                # print( self.g_cumulative_max_dict )
                # raise exc
                cumulative_num_steps    = 0

        if info.step >= 0:
            self.stream_cur_dict[info.stream]    = cumulative_num_steps + max( 0, info.step )
        self.stream_state_dict[info.stream]  = state
        self.current_stream = info.stream

        return info

    def get_stream_states( self ):
        return [ self.stream_state_dict[stream]
                    for stream in range( len(self.stream_state_dict) ) ]

    def get_stream_progresses( self ):
        return [ ( self.stream_max_dict[stream], self.stream_cur_dict[stream], self.stream_state_dict[stream] )
                    for stream in range( len(self.stream_state_dict) ) ]

    def is_end_states( self, states ):
        is_end = True
        for stream, state in self.stream_state_dict.items():
            max_ = self.stream_max_dict[stream]
            if max_ == 0:
                # if there is nothing todo, it is ok to be regarded as done
                continue

            if state == STATE_INIT or state == STATE_DOING:
                is_end = False
            elif state == STATE_ERROR:
                cur_ = self.stream_cur_dict[stream]
                # print( 'cur_, max_', cur_, max_ )
                if cur_ < max_:
                    # In cases of STATE_ERROR
                    # it is not taken as finished if there remain things to do.
                    # On the other hand, STATE_FATAL means imediately "is_end"
                    is_end = False
        return is_end

    def set_error( self ):
        put_error( self.current_stream )

def put_info( k, step, state=None ):
    stream, phase = k
    info = ExecInfo( info_type=INFO_TYPE_STEP, stream=stream, phase=phase, step=step, state=state )
    exe_queue.put( info )

def put_error( k, step=-1, error_state=STATE_ERROR ):
    if type( k ) == tuple:
        stream, phase = k
    else:
        stream = k
        phase  = -1
    info = ExecInfo( info_type=INFO_TYPE_STEP, stream=stream, phase=phase, step=step, state=error_state )
    exe_queue.put( info )

def put_max_info( stream, max_dict ):
    # check whether max_dict is restricted to stream
    for k, v in max_dict.items():
        assert( k[0] == stream )

    info = ExecInfo( info_type=INFO_TYPE_MAX, stream=stream, max_dict=max_dict )
    exe_queue.put( info )

def send_stop():
    cmd_queue.put( 'STOP' )

def on_stop_raise( cleanup=None, log_closure=None ):
    try:
        cmd = cmd_queue.get( False )
        if log_closure is not None:
            log_closure(cmd)
    except:
        cmd = None

    if cmd is not None:
        if cleanup is not None:
            cleanup()
        raise Exception( 'STOP' )
