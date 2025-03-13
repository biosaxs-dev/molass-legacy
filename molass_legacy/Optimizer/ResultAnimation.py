"""
    Optimizer.ResultAnimation.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
from time import sleep
from matplotlib.animation import ArtistAnimation
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

def show_result_animation_impl(selector_dialog, debug=True):
    optimizer_folder = get_setting('optimizer_folder')
    joblist_folder = selector_dialog.folder.get()
    print("show_result_animation_impl: optimizer_folder=", optimizer_folder)
    print("show_result_animation_impl: joblist_folder=", joblist_folder)

    nodes = []
    for i, node in enumerate(os.listdir(joblist_folder)):
        # print([i], node)
        nodes.append(node)

    def plot_a_job(anim_ax, axis_info, node, joblist_folder, title_text):
        fig, axes = axis_info
        jobpath = os.path.join(joblist_folder, node)
        set_setting('optworking_folder', jobpath)
        result = selector_dialog.get_result(folder=jobpath)
        optimizer = result.get_optimizer()
        ret_ims = []
        for i, params in result.get_result_iterator():
            for ax in axes:
                ax.cla()
            axes[-1].grid(False)
            title_text.set_text("Result Animation of %s/%s-%03d" % (joblist_folder, node, i))
            optimizer.objective_func(params, plot=True, axis_info=axis_info)
            fig.tight_layout()
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            im = anim_ax.imshow(buf)
            ret_ims.append([im])
        return ret_ims

    with plt.Dp(window_title="Result Animation", button_spec=["OK", "Cancel"]):
        anim_fig, anim_ax = plt.subplots(figsize=(18,5))
        anim_ax.set_axis_off()

        with plt.Dp():
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
            title_text = fig.suptitle("Result Animation of %s" % joblist_folder, fontsize=20)
            axt = ax2.twinx()
            axis_info= (fig, (ax1, ax2, ax3, axt))

            images = []
            for node in nodes:
                ims = plot_a_job(anim_ax, axis_info, node, joblist_folder, title_text)
                images += ims

        anim = ArtistAnimation(anim_fig, images, interval=200, blit=True,
                                repeat_delay=1000)
    
        anim.save(os.path.join(optimizer_folder, "result_animation.gif"))
        plt.show()