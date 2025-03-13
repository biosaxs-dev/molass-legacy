"""

    PopupMenuUtils.py

    Copyright (c) 2024, Masatsuyo Takahashi, KEK-PF

"""
def post_popup_menu(popup_menu, anchor_widget, event, mpl_event=False, debug=False):
    from molass_legacy.KekLib.TkUtils import split_geometry

    cx = anchor_widget.winfo_rootx()
    cy = anchor_widget.winfo_rooty()
    w, h, x, y = split_geometry(anchor_widget.winfo_geometry())

    if debug:
        print("cx, cy=", cx, cy)
        print("w, h, x, y=", w, h, x, y)
        print("event.x, event.y=", event.x, event.y)

    if mpl_event:
        popup_menu.post(cx + int(event.x), cy + h - int(event.y))
    else:
        # currently adjusted only for VpaButtonFrame
        # why + 100, +20 ?
        popup_menu.post(cx + x + int(event.x) + 100, cy + int(event.y) + 20)

if __name__ == '__main__':
    pass
