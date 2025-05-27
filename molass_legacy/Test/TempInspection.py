"""
    TempInspection.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
from time import sleep
from DataUtils import get_in_folder
from molass_legacy.KekLib.BasicUtils import clear_dirs_with_retry

fh = None

def datarange_problem_inspection(agent, result_folder):
    global fh

    in_folder = get_in_folder(agent.in_folder.get())
    print("\n\n\n------------------- datarange_problem_inspection", in_folder)
    print("result_folder=", result_folder)

    agent.show_datarange_dialog(__wait__=False)

    while not agent.has_datarange_dialog():
        print("waiting for the datarange_dialog being ready")
        sleep(1)

    dialog = agent.datarange_dialog

    xr_frame = dialog.get_current_frame()
    xr_frame.toggle_btn.invoke()
    sleep(1)

    uv_frame = dialog.get_current_frame()
    # canvases[0]: class_3d
    # canvases[1]: class_2d - upper right frame
    # canvases[2]: class_2d - lower right frame
    uv_frame.canvases[1].from_opposite_side_btn.invoke()

    dialog.cancel()
    print("------------------- canceled")
    sleep(1)

    if fh is None:
        # clear_dirs_with_retr([result_folder])
        path = os.path.join(result_folder, "inspected.csv")
        fh = open(path, "w")
        fh.write("folder\n")
    fh.write(",".join([in_folder]) + "\n")

    agent.analysis_button.invoke(__wait__=False)
    sleep(1)
    analyzer = agent.analyzer

    while not analyzer.has_mapper_canvas():
        sleep( 1 )

    analyzer.mapper_canvas.cancel()
    sleep( 1 )

def inspection_close(agent):
    fh.close()
    print("------------------- closed\n\n\n")
