"""
KekLib.IpyUtils.py - Utility functions for interactive user prompts in Jupyter notebooks.
"""
import ipywidgets as widgets
from IPython.display import display, clear_output

def ask_user(question, callback=None, output_widget=None):
    """
    Display a Yes/No dialog in a dedicated Output widget.

    Parameters:
    -----------
    question : str
        The question to present to the user.
    callback : function, optional
        A function to call with the user's response (True for Yes, False for No).
    output_widget : ipywidgets.Output, optional
        An existing Output widget to use for displaying the dialog. If None, a new one is created.
        
    Returns:
    --------
    output_widget : ipywidgets.Output
    """
    if output_widget is None:
        output_widget = widgets.Output()
        display(output_widget)  # Only display if not already displayed

    with output_widget:
        clear_output(wait=True)
        label = widgets.Label(question)
        btn_yes = widgets.Button(description="Yes", button_style='success')
        btn_no = widgets.Button(description="No", button_style='danger')
        button_box = widgets.HBox([btn_yes, btn_no])
        display(widgets.VBox([label, button_box]))

        def on_yes(b):
            with output_widget:
                clear_output(wait=True)
                print("You selected: Yes")
            if callback:
                callback(True)

        def on_no(b):
            with output_widget:
                clear_output(wait=True)
                print("You selected: No")
            if callback:
                callback(False)

        btn_yes.on_click(on_yes)
        btn_no.on_click(on_no)

    return output_widget