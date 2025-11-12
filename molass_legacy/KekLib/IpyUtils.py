"""
KekLib.IpyUtils.py - Utility functions for interactive user prompts in Jupyter notebooks.
"""
import ipywidgets as widgets
from IPython.display import display, clear_output

def ask_user(dialog_output, question, callback):
    """
    Ask a yes/no question to the user in a Jupyter notebook and execute a callback with the response.

    Parameters
    ----------
    question: str
        The question to ask the user.
    callback: function
        A function that takes a single boolean argument (True for 'Yes', False for 'No').
    
    Example
    -------
    response = None

    # Usage example:
    def handle_response(answer):
        global response
        response = answer
        print("Callback received:", answer)

    ask_user("Do you want to continue?", handle_response)
    """
    btn_yes = widgets.Button(description="Yes")
    btn_no = widgets.Button(description="No")
    out = widgets.Output()

    def on_yes(b):
        clear_output()
        print("You selected: Yes")
        callback(True)

    def on_no(b):
        clear_output()
        print("You selected: No")
        callback(False)

    btn_yes.on_click(on_yes)
    btn_no.on_click(on_no)

    with dialog_output:
        clear_output(wait=True)
        display(widgets.VBox([widgets.Label(question), widgets.HBox([btn_yes, btn_no])]), out)