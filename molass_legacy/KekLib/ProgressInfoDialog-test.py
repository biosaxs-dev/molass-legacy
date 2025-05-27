"""

    ProgressInfoDialog.py

    Copyright (c) 2017-2023, Masatsuyo Takahashi, KEK-PF

"""

import os
import re
import warnings
import time
import logging
from io import StringIO

from molass_legacy.KekLib.BasicUtils             import get_caller_module
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog, ttk, Font, ScrolledText
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable, BlinkingFrame, FLASH_STOP_JUST
from molass_legacy.KekLib.TkUtils                import split_geometry, join_geometry
import OurMessageBox        as MessageBox
from Settings               import get_setting
from ProgressInfo           import STATE_INIT, STATE_DOING, STATE_DONE, STATE_ERROR, STATE_FATAL, SAFE_FINISH, cleanup_queues

COLUMN_TITLES   = [ 'stage', 'bar', 'time', 'todo', 'done', 'state' ]
TEXT_TAGS       = { logging.ERROR:'ERROR_TAG', logging.WARNING:'WARNING_TAG' }
TAG_COLORS      = { 'ERROR_TAG':'red', 'WARNING_TAG':'orange' }
STATE_TEXTS     = {
    STATE_INIT  : '',
    STATE_DOING : 'doing',
    STATE_DONE  : 'done',
    STATE_ERROR : 'error',
    STATE_FATAL : 'error',
    }
GREEN   = 'green4'

def make_disp_time( t_ ):
    m_ = t_ // 60
    s_ = t_ % 60
    disp_time = '%2d′%2d″' % ( m_, s_ )
    return disp_time

class ProgressLoghandler( logging.Handler ):
    def __init__( self, log_records ):
        logging.Handler.__init__( self )
        self.log_records  = log_records

    def emit( self, record ):
        if record.levelno >= logging.INFO:
            self.log_records.append( record )

class ProgressInfoDialog( Dialog ):
    def __init__( self, parent, title, pinfo, description_cb=None, stream_labels=None,
                    plot_init_cb=None, plot_update_cb=None, refresh_log_cb=None,
                    plot_final_cb=None,
                    is_alive_cb=None,
                    refresh_interval=100,
                    logger=None ):
        self.grab       = 'local'     # used in grab_set
        self.parent = parent
        self.title_ = title
        self.pinfo  = pinfo
        self.description_cb = description_cb
        self.plot_init_cb   = plot_init_cb
        self.plot_update_cb = plot_update_cb
        self.refresh_log_cb = refresh_log_cb
        self.plot_final_cb  = plot_final_cb
        self.is_alive_cb = is_alive_cb
        self.worker_is_alive = True
        self.stream_labels  = stream_labels
        self.fixed_font = Font.Font( family="Courier", size=9 )
        self.stop_mode  = False
        self.caller_module = get_caller_module( level=2 )
        self.is_at_end_ = False
        self.refresh_interval = refresh_interval
        self.logger = logger
        Dialog.__init__( self, self.parent, self.title_, visible=False )

    def show( self, terminate_cb=None, logger=None ):
        cleanup_queues()        # just in case when there remain previous messages
        self.show_return = None
        self.terminate_cb = terminate_cb
        self.log_records = []
        if logger is not None:
            log_handler = ProgressLoghandler( self.log_records )
            logger.addHandler( log_handler )

        # need to use parent
        # because self is not yet a tk widget before the __init__ call
        self.monitoring_loop = True
        self.can_safely_finish = False
        self.parent.after( self.refresh_interval, self.refresh )
        self.stream_start_time = time.time()
        self._show()

        if logger is not None:
            logger.removeHandler( log_handler )
            log_handler = None

        return self.show_return

    def body( self, body_frame ):
        tk_set_icon_portable( self, module=self.caller_module )

        self.vars_ = []
        self.bars  = []
        self.blinks = []

        iframe = Tk.Frame( body_frame )
        iframe.pack( padx=40 )

        if self.description_cb is not None:
            self.description_cb( self, iframe )

        progress_frame  = Tk.Frame( iframe )
        progress_frame.pack( fill=Tk.X, pady=10 )
        progress_label  = Tk.Label( progress_frame, text="Progress", width=12, anchor=Tk.W )
        progress_label.grid( row=0, column=0, sticky=Tk.N + Tk.W )
        progress_table  = Tk.Frame( progress_frame )
        progress_table.grid( row=0, column=1, sticky=Tk.E )

        for j, col_title in enumerate( COLUMN_TITLES ):
            label   = Tk.Label( progress_table, text=col_title )
            label.grid( row=0, column=j, padx=5 )

        row = 0
        self.first_stream = None
        self.num_buttons = 0
        self.button_states = []
        for i in range( len( self.pinfo.stream_max_dict ) ):
            tim_var = Tk.StringVar()
            max_var = Tk.StringVar()
            val_var = Tk.StringVar()
            stt_var = Tk.StringVar()
            self.vars_.append( [ tim_var, max_var, val_var, stt_var ] )
            self.button_states.append(0)
            self.num_buttons += 1
            stream_size = self.pinfo.stream_max_dict[i]

            bar_label = 'Bar ' + str(i) if self.stream_labels is None else self.stream_labels[i]
            label   = Tk.Label( progress_table, text=bar_label, font=self.fixed_font )
            bar = ttk.Progressbar( progress_table, orient='horizontal', length=200, mode='determinate' )
            self.bars.append( bar )
            state_label = Tk.Label( progress_table, textvariable=stt_var, justify=Tk.LEFT, fg=GREEN )
            objects = []
            objects.append( [ state_label, { 'row':row+1, 'column':5, 'sticky':Tk.W } ] )
                                        # row, column is based on this parent iframe1
            blink = BlinkingFrame( progress_table, objects, grid=True, flash_stop_type=FLASH_STOP_JUST )
            self.blinks.append( [ state_label, blink ] )

            if stream_size == 0:
                # no grid call is done while the objects exist
                continue

            if self.first_stream is None:
                self.first_stream = i
            row += 1
            label.grid( row=row, column=0, padx=5 )
            bar.grid( row=row, column=1, padx=5 )
            text    = Tk.Label( progress_table, textvariable=tim_var, justify=Tk.RIGHT )
            text.grid( row=row, column=2, padx=5 )
            text    = Tk.Label( progress_table, textvariable=max_var, justify=Tk.RIGHT )
            text.grid( row=row, column=3, padx=5 )
            text    = Tk.Label( progress_table, textvariable=val_var, justify=Tk.RIGHT )
            text.grid( row=row, column=4, padx=5 )
            blink.grid( row=row, column=5 )

        row += 1
        self.total_tim_lbl = Tk.StringVar()
        label   = Tk.Label( progress_table, textvariable=self.total_tim_lbl, justify=Tk.RIGHT )
        label.grid( row=row, column=1, sticky=Tk.E )
        self.total_tim_var = Tk.StringVar()
        text    = Tk.Label( progress_table, textvariable=self.total_tim_var, justify=Tk.RIGHT )
        text.grid( row=row, column=2, padx=5 )

        if self.plot_init_cb is not None:
            self.plot_init_cb( self, iframe )

        log_frame = Tk.Frame( iframe )
        log_frame.pack( fill=Tk.BOTH, expand=1 )
        label = Tk.Label( log_frame, text="Execution Log" )
        label.pack( anchor=Tk.NW )

        self.log_text = ScrolledText( log_frame, width=100, height=20 )
        self.log_text.pack( fill=Tk.BOTH )
        for tag, color in TAG_COLORS.items():
            self.log_text.tag_configure( tag, foreground=color )

    def buttonbox(self):
        '''
        override standard buttonbox.
        add "Default" button
        '''

        box = Tk.Frame(self)

        self.button = Tk.Button(box, text="Cancel", width=10, command=self.button_command)
        self.button.pack(side=Tk.LEFT, padx=5, pady=10)

        self.bind("<Escape>", self.button_command)
        box.pack()

    def cancel( self ):
        """
            We have to override Dialog.cancel because the following
            call is implemented in Dialog.__init__

            self.protocol( "WM_DELETE_WINDOW", self.cancel )
        """
        self.button_command()

    def button_command( self, *args ):
        btn_text = self.button.cget( 'text' )
        if btn_text == 'OK':
            self.terminate( btn_text )
            return

        yn = MessageBox.askyesno(
                'Cancel Confirmation',
                'Do you really want to stop the execution?',
                parent=self,
                )
        if yn:
            self.stop_mode = True
            parent_ = self.parent
            self.terminate( btn_text )
            MessageBox.showinfo(
                'Cancel Notification',
                'Process has been canceled.',
                parent=parent_,
                )

    def terminate( self, btn_text ):
        if self.terminate_cb is not None:
            self.terminate_cb( btn_text )
        self.show_return = self.button.cget( 'text' )
        self.destroy()

    def get_log_text(self, fmt='%(asctime)s %(message)s', datefmt='%H:%M:%S'):
        """ Convert log messages to text. On formats see logger.Formatter """

        if len( self.log_records ) == 0:
            return None

        f   = logging.Formatter(fmt, datefmt)
        buf = StringIO()
        b   = 0
        warn_stat = 0
        warn_end = 0
        max_level = None

        while True:
            try:
                record = self.log_records.pop(0)
            except IndexError:
                # stop if there exist no records left
                break

            if max_level is None or record.levelno > max_level:
                max_level = record.levelno
            if b:
                b += buf.write('\n')
                if record.levelno >= logging.WARNING:
                    if warn_stat == 0:
                        warn_stat = b
                    warn_end = b

            try:
                formated_record = f.format(record)
            except:
                from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
                etb = ExceptionTracebacker()
                formated_record = 'format failed with %s "%s"\n' % (etb.last_lines(), str(record))
            b += buf.write(formated_record)
            if max_level >= logging.WARNING:
                warn_end = b
                # stop once in order to avoid coloring of normal messages
                break

        text = buf.getvalue()
        buf.close()

        return text, max_level, warn_stat, warn_end

    def refresh_log( self ):
        ret = self.get_log_text()
        if ret is None:
            return

        text, level, warn_stat, warn_end = ret
        # print( 'warn_stat, warn_end=', warn_stat, warn_end )

        if self.refresh_log_cb is not None:
            self.refresh_log_cb( text, level )

        for key_word, key_level in [    [ 'error',      logging.ERROR ],
                                        [ 'warning',    logging.WARNING ],
                                        ]:
            if text.lower().find( key_word ) >= 0:
                level = max( level, key_level )

        if level >= logging.ERROR:
            self.pinfo.set_error()

        has_error_or_warinng = level >= logging.WARNING
        log_text = self.log_text
        log_text.configure( state=Tk.NORMAL )

        if has_error_or_warinng:
            # reduce the index by one, e.g., 13.0 ==> 12.0
            msg_start = re.sub( r'(\d+).', lambda m: '%d.' % ( int( m.group(1) ) - 1 ), log_text.index( Tk.END ) )

        log_text.insert( Tk.END, text )

        if has_error_or_warinng:
            msg_end = log_text.index( Tk.END )

        log_text.insert( Tk.END, '\n' )

        if has_error_or_warinng:
            tag = TEXT_TAGS[level]
            if warn_stat > 0:
                # modify msg_start, msg_end
                t1 = text[warn_stat:warn_end]
                t2 = text[warn_end:]
                n_ = [ len( [pos for pos, char in enumerate(t_) if char == '\n'] ) for t_ in [t1, t2 ] ]
                n1 = n_[0] + n_[1]
                n2 = n_[1]
                msg_start = re.sub( r'(\d+).', lambda m: '%d.' % ( int( m.group(1) ) - (n1+1) ), msg_end )
                if n2 > 0:
                    msg_end = re.sub( r'(\d+).', lambda m: '%d.' % ( int( m.group(1) ) - (n2+1) ), msg_end )
                # print( 'n_=', n_, 'msg_start, msg_end=', (msg_start, msg_end) )
            log_text.tag_add( tag, msg_start, msg_end )

        log_text.see( Tk.END )

        log_text.configure( state=Tk.DISABLED )

    def refresh( self, terminate_loop=False ):
        self.refresh_log()

        info = self.pinfo.get_info()

        current_stream = self.pinfo.current_stream
        todo = self.pinfo.stream_max_dict[current_stream]
        if todo == 0:
            # update time only for todo > 0
            self.after( self.refresh_interval, self.refresh )
            return

        start_time = self.pinfo.get_start_time(current_stream)
        this_time = time.time()
        if start_time is None:
            disp_time = ''
        else:
            disp_time = make_disp_time( this_time - start_time )
        self.vars_[current_stream][0].set( disp_time )
        if current_stream > self.first_stream:
            start_time = self.pinfo.get_start_time(self.first_stream)
            if start_time is not None:
                self.total_tim_lbl.set( 'cumulative time' )
                self.total_tim_var.set( make_disp_time( this_time - start_time ) )

        if info is None:
            num_stopped = 0
            for state in self.button_states:
                num_stopped += state
            if self.num_buttons > num_stopped:  # namely, if there remains an un-finished button
                self.after( self.refresh_interval, self.refresh )
            else:
                if not self.can_safely_finish:
                    self.after( self.refresh_interval, self.refresh )
                # better termination test desired including minor error cases
                pass
            return

        progresses = self.pinfo.get_stream_progresses()

        if False:
            logger = logging.getLogger( __name__ )
            logger.info( 'progresses=' + str(progresses) )

        for i, var_row in enumerate( self.vars_ ):
            _, max_var, val_var, stt_var = var_row
            progress = progresses[i]
            if progress[0]  == 0:
                # if todo == 0 then skip
                continue

            max_var.set( str( progress[0] ) )
            val_var.set( str( progress[1] ) )
            state = progress[2]
            stt_var.set( STATE_TEXTS[state] )

            label   = self.blinks[i][0]
            blink   = self.blinks[i][1]
            if state == STATE_DOING:
                if not blink.switch:
                    blink.start()

                if label.cget('fg') == 'red':
                    label.config( fg=GREEN )
                elif label.cget('bg') == 'red':
                    label.config( bg=GREEN )

                if self.is_alive_cb is not None:
                    is_alive = self.is_alive_cb()
                    is_alive = False
                    if self.worker_is_alive and not is_alive:
                        self.found_inactive_time = time.time()
                        self.worker_is_alive = False
                    if not is_alive:
                        t = time.time() - self.found_inactive_time
                        if t > 10:
                            state = STATE_DONE
                            if self.logger is None:
                                logger = logging.getLogger(__name__)
                            else:
                                logger = self.logger
                            logger.warning("state has been forced to be STATE_DONE after %.3g inactive time", t)

            if state in [ STATE_DONE, STATE_ERROR, STATE_FATAL ]:
                blink.stop()
                self.button_states[i] = 1       # this terminates the after loop when all the button states are 1

                if state == STATE_DONE:
                    label.config( bg=GREEN )
                    label.config( fg='white' )
                else:
                    label.config( bg='red' )
                    label.config( fg='white' )
                label.update()

            elif state == SAFE_FINISH:
                self.can_safely_finish = True

            bar = self.bars[i]
            bar['maximum']  = progress[0]
            bar['value']    = min( progress[0], progress[1] )
        states = self.pinfo.get_stream_states()

        if self.plot_update_cb is not None:
            self.plot_update_cb( self )

        if self.pinfo.is_end_states( states ):
            self.is_at_end_ = True
            self.button.config( text='OK' )
            self.bind("<Return>", self.button_command)
            if self.plot_final_cb is not None:
                self.plot_final_cb()
            terminate_next = True
            # there exist remaining log texts which have not yet been shown
        else:
            terminate_next = False

        if not terminate_loop:
            self.after( self.refresh_interval, lambda: self.refresh(terminate_loop=terminate_next) )

    def is_at_end( self ):
        return self.is_at_end_
    