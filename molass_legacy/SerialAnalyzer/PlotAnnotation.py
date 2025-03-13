# coding: utf-8
"""

    PlotAnnotation.py

        recognition of peaks

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF

"""

def add_flow_change_annotation( ax, flow_changes, mapper ):
    xmin1, xmax1 = ax.get_xlim()
    ymin1, ymax1 = ax.get_ylim()
    yoffset = ( ymax1 - ymin1 )*0.2
    for i, fc in enumerate( flow_changes ):
        if fc is None:
            continue
        xoffset = ( xmax1 - xmin1 )*( 0.05 if fc < 100 else 0 )
        y_ = mapper.a_spline( fc )
        ax.annotate( "flow change", xy=(fc, y_),
                        xytext=( fc + xoffset, y_ + yoffset ),
                        ha='center',
                        arrowprops=dict( headwidth=5, width=0.5, color='black', shrink=0.05 ),
                        )
