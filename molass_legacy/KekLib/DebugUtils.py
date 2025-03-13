"""
    DebugUtils.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import os
from termcolor import colored

os.system('color')  # from https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal

last_lines = None

def show_call_stack(line_prefix, stop_level=None, indented_lines=6, indented_only=False):
    """
    for usage, see Trimming.AutoRestrictor.py where this is used for the first time
    """
    import inspect
    global last_lines

    stack = inspect.stack()[1:stop_level]

    if not indented_only:
        for frm in stack:
            print("%s%s %s (%d)" % (line_prefix, frm.filename, frm.function, frm.lineno))

    n = indented_lines
    stored_lines = []
    for k in range(n):
        i = n - k - 1
        frm = stack[i]
        line = "%s%s (%d)" % (" "*(4*k), frm.function, frm.lineno)
        stored_lines.append(line)
        color_print = False
        if last_lines is not None:
            if k < len(last_lines):
                if line != last_lines[k]:
                    color_print = True

        if color_print:
            print(colored(line, 'yellow'))
        else:
            print(line)

    last_lines = stored_lines
