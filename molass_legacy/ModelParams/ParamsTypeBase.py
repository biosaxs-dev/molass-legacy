"""
    ModelParams.ParamsTypeBase.py

    Abstract base class for all parameter-layout managers (*Params classes).

    When implementing a new elution model, inherit from this class to get
    an early ImportError if required GUI methods are missing, rather than
    discovering the gap at GUI runtime.

    See molass-legacy issue #81.

    Copyright (c) 2026, SAXS Team, KEK-PF
"""


class ParamsTypeBase:
    """Base class that documents the required interface for *Params classes.

    GUI callers expect:
    - ``get_params_sheet(parent, params, dsets, optimizer)``
        Returns the parameter inspection sheet used by the "Show Parameters"
        button (``ParamsInspection``, ``ParamsSelector``, ``LrfExporter``).
    - ``get_paramslider_info()``
        Returns slider metadata used by the "Parameter Slider" button.

    New model implementations should inherit from this class.
    Existing classes are not required to inherit it immediately, but should
    be migrated over time.
    """

    def get_params_sheet(self, parent, params, dsets, optimizer, debug=True):
        """Return the parameter inspection sheet for the GUI 'Show Parameters' button.

        Must be overridden by subclasses.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_params_sheet(). "
            "See ModelParams/ParamsTypeBase.py for the required interface."
        )

    def get_paramslider_info(self, devel=True):
        """Return slider metadata for the GUI 'Parameter Slider' button.

        Must be overridden by subclasses.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_paramslider_info(). "
            "See ModelParams/ParamsTypeBase.py for the required interface."
        )
