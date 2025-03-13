# -*- coding: utf-8 -*-
import sys

if sys.version_info > (3,):
    from unittest.mock import MagicMock
else:
    from mock import MagicMock
