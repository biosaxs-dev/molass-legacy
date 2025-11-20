"""
KekLib.IpyLabelUtils.py
"""

from IPython.display import display, HTML

def inject_label_color_css():
    """
    This function adds CSS styles for different label colors (green, red, yellow, gray)
    to be used with ipywidgets.Label.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """

    display(HTML("""
    <style>
    .status-label-green {
        color: green !important;
        font-weight: bold;
    }
    .status-label-red {
        color: red !important;
        font-weight: bold;
    }
    .status-label-yellow {
        color: orange !important;
        font-weight: bold;
    }
    .status-label-yellow {
        color: orange !important;
        font-weight: bold;
    }
    .status-label-gray {
        color: gray !important;
        font-weight: bold;
    }
    .status-label-blue {
        color: blue !important;     
        font-weight: bold;
    }
    </style>
    """))

def set_label_color(label, color):
    """
    Set the color of an ipywidgets.Label by adding/removing CSS classes.

    Parameters:
    -----------
    label : ipywidgets.Label
        The label widget to modify.
    color : str
        The color to set. Must be one of: "green", "red", "yellow", "gray", "blue".
        
    Returns:
    --------
    None
    """
    if color not in ["green", "red", "yellow", "gray", "blue"]:
        raise ValueError("Color must be one of: green, red, yellow, gray, blue.")
    
    # Remove all color classes
    for c in ["status-label-green", "status-label-red", "status-label-yellow", "status-label-gray", "status-label-blue"]:
        label.remove_class(c)
    # Add the selected color class
    label.add_class(f"status-label-{color}")