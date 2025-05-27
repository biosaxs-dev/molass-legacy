# -*- coding: utf-8 -*-
"""

    ファイル名：   GuiPreferences.py

    処理内容：

        GUI のオプション設定情報

"""
from __future__ import division, print_function, unicode_literals

from molass_legacy.KekLib.OurTkinter     import Tk, Dialog, ttk, tk_set_icon_portable
import OurMessageBox    as MessageBox
from Preferences    import get_preference, set_preference, save_preferences

class PreferencesDialog( Dialog ):

    def __init__( self, parent, title, pilatus_counter, action=0 ):
        self.grab = 'local'     # used in grab_set
        self.pilatus_counter    = pilatus_counter
        self.action             = action
        self.applied            = False
        self.couter_id_changed  = False
        self.adj_output_changed = False
        self.trace_             = True

        Dialog.__init__(self, parent, title) # this calls body

    def body( self, body_frame ):   # overrides parent class method

        tk_set_icon_portable( self, 'synthesizer' )

        iframe = Tk.Frame( body_frame );
        iframe.pack( expand=1, fill=Tk.BOTH, padx=10, pady=10 )

        label_spacing       = '     '
        text_entry_width    = 12

        grid_row = -1

        if self.action in [ 0, 3 ]:
            grid_row += 1
            syn_method_label = Tk.Label( iframe, text= 'Sythesizing Method:' + label_spacing )
            syn_method_label.grid( row=grid_row, column=0, sticky=Tk.E )
            self.syn_method = Tk.StringVar()
            self.syn_method.set( get_preference( 'syn_method' ) )
            j = 0
            for t, method in ( [ 'Just Cover', 'cover' ], [ 'Compute Average', 'average']  ):
                b = Tk.Radiobutton( iframe, text='%-22.22s' % (t),
                                variable=self.syn_method, value=method )
                j += 1
                b.grid( row=grid_row, column=j, sticky=Tk.W )

            grid_row += 1
            self.detection_counter = Tk.StringVar()
            self.detection_counter.set( get_preference( 'detection_counter' ) )
            detection_counter_label = Tk.Label( iframe, text= 'Beam Intensity Counter:' + label_spacing )
            detection_counter_label.grid( row=grid_row, column=0, sticky=Tk.E )
            detection_counter_cbox = ttk.Combobox( iframe, textvariable=self.detection_counter, width=text_entry_width, justify=Tk.CENTER )
            values = [ 'None' ]
            # TODO: make sure that self.pilatus_counter is not None
            #       and handle the situation where 'detection_counter' is not available
            if self.pilatus_counter:
                for counter in self.pilatus_counter.available_counters():
                    values.append( counter )
            detection_counter_cbox['values'] = values
            detection_counter_cbox.grid( row=grid_row, column=1, sticky=Tk.W )

            grid_row += 1
            self.postfix_syn = Tk.StringVar()
            self.postfix_syn.set( get_preference( 'postfix_syn' ) )
            postfix_syn_label = Tk.Label( iframe, text= 'Synthesized File Postfix:' + label_spacing )
            postfix_syn_label.grid( row=grid_row, column=0, sticky=Tk.E )
            postfix_syn_entry = Tk.Entry( iframe, textvariable=self.postfix_syn, width=text_entry_width, justify=Tk.CENTER )
            postfix_syn_entry.grid( row=grid_row, column=1, sticky=Tk.W )

        if self.action in [ 0, 1, 2 ]:

            grid_row += 1
            color_map_label = Tk.Label( iframe, text= 'Color Mapping Scheme:' + label_spacing )
            color_map_label.grid( row=grid_row, column=0, sticky=Tk.E )

            self.color_map = Tk.StringVar()
            self.color_map.set( get_preference( 'color_map' ) )
            j = 0
            for t, n in ( [ 'ALBULA Like', 'ALBULA' ], [ 'Diverging', 'Diverging' ] ):
                b = Tk.Radiobutton( iframe, text=t,
                                variable=self.color_map, value=n )
                j += 1
                b.grid( row=grid_row, column=j, sticky=Tk.W )

        if self.action in [ 3 ]:
            grid_row += 1
            syn_images_label = Tk.Label( iframe, text= 'Image Files to Synthesize:' + label_spacing )
            syn_images_label.grid( row=grid_row, column=0, sticky=Tk.E )

            self.syn_policy = Tk.StringVar()
            self.syn_policy.set( get_preference( 'syn_policy' ) )
            j = 0
            for t, policy in ( [ 'All', 'all' ], [ 'Select ⇒', 'select']  ):
                b = Tk.Radiobutton( iframe, text=t,
                                variable=self.syn_policy, value=policy )
                b.grid( row=grid_row+j, column=1, sticky=Tk.W )
                j += 1

            if self.syn_policy.get() == 'all':
                state = 'disabled'
            else:
                state = 'normal'

            grid_row += 1
            flags = get_preference( 'syn_flags' )
            self.image_syn_flags = []
            cbs = []
            j = 0
            for text in ( '*_0_*.tif', '*_1_*.tif', '*_2_*.tif' ):
                ivar = Tk.IntVar()
                ivar.set( flags[j] )
                self.image_syn_flags.append( ivar )
                state_ = state
                if j == 0:
                    state_ = 'disabled'         # 当面原画は必須（変更不可）とする。
                b = Tk.Checkbutton( iframe, text='%-22.22s' % (text), variable=ivar, state=state_ )
                cbs.append( b )
                b.grid( row=grid_row, column=2+j, sticky=Tk.W )
                j += 1

            def syn_policy_trace( *args ):
                policy = self.syn_policy.get()
                if policy == 'all':
                    state = 'disabled'
                else:
                    state = 'normal'
                j = 0
                for b in cbs:
                    if policy == 'all':
                        self.image_syn_flags[j].set( 1 )
                    state_ = state
                    if j == 0:
                        state_ = 'disabled'     # 当面原画は必須（変更不可）とする。
                    b.config( state=state_ )
                    j += 1

            self.syn_policy.trace( "w", syn_policy_trace )

            self.trace_  = True

            def syn_select_image_trace( *args ):
                if not self.trace_:
                    # 以下のチェック後の処置中の呼び出しを無視する。
                    # なぜか UnboundLocalError: local variable 'trace_' referenced before assignment
                    # のためメンバー変数とした。
                    return

                count = 0
                for ivar in self.image_syn_flags:
                    if ivar.get() == 1:
                        count += 1
                if count < 2:
                    MessageBox.showinfo( 'Select More', 'You must select at least two.' )
                    self.trace_ = False
                    self.syn_policy.set( 'all' )
                    self.trace_ = True

            for ivar in self.image_syn_flags:
                ivar.trace( "w", syn_select_image_trace )

        if self.action in [ 0 ]:
            grid_row += 1
            save_policy_label = Tk.Label( iframe, text= 'Foler/File Entries Save Policy:' + label_spacing )
            save_policy_label.grid( row=grid_row, column=0, sticky=Tk.E )
            self.save_policy = Tk.StringVar()
            self.save_policy.set( get_preference( 'save_policy' ) )
            j = 0
            for t, n in ( [ 'Ask', 'Ask' ], [ 'Save', 'Yes' ], [ 'No Save', 'No' ] ):
                b = Tk.Radiobutton( iframe, text=t,
                                variable=self.save_policy, value=n )
                j += 1
                b.grid( row=grid_row, column=j, sticky=Tk.W )

        # global grab cannot be set befor windows is 'viewable'
        # and this happen in mainloop after this function returns
        # Thus, it is needed to delay grab setting of an interval
        # long enough to make sure that the window has been made
        # 'viewable'
        if self.grab == 'global':
            self.after(100, self.grab_set_global )
        else:
            pass # local grab is set by parent class constructor

    def apply( self ):  # overrides parent class method
        print( "ok. apply" )

        if self.action in [ 0, 3 ]:
            set_preference( 'syn_method', self.syn_method.get() )
            old_counter_id = get_preference( 'detection_counter' )
            new_counter_id = self.detection_counter.get()
            if new_counter_id != old_counter_id:
                self.couter_id_changed = True
            set_preference( 'detection_counter', new_counter_id )
            set_preference( 'postfix_syn', self.postfix_syn.get() )

        if self.action in [ 0, 1, 2 ]:
            set_preference( 'color_map', self.color_map.get() )

        if self.action in [ 3 ]:
            flags = []
            for ivar in self.image_syn_flags:
                flags.append( ivar.get() )
            # print( 'self.image_syn_flags=', flags )
            set_preference( 'syn_flags', flags )

        if self.action in [ 0 ]:
            set_preference( 'save_policy', self.save_policy.get() )
            save_preferences()

        self.applied    = True
