"""
    Simplified version of multiprocessin.shared_memory.SharedMemory
    for debug purposes
"""

import mmap
from .OurMultiprocessing import _make_filename

ANONYMOUS_MEMORY = -1

class SharedMemory:
    def __init__(self, name=None, create=False, size=0):
        if create:
            temp_name = _make_filename() if name is None else name
            self._mmap = mmap.mmap(ANONYMOUS_MEMORY, size, tagname=temp_name)
            self._name = temp_name
        else:
            self._mmap = mmap.mmap(ANONYMOUS_MEMORY, size, tagname=name)
            self._name = name

        self._size = size
        self._buf = memoryview(self._mmap)

    def __del__(self):
        try:
            self.close()
        except OSError:
            pass

    def close(self):
        # print(__name__, 'close')
        """Closes access to the shared memory from this instance but does
        not destroy the shared memory block."""
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    @property
    def buf(self):
        "A memoryview of contents of the shared memory block."
        return self._buf

    @property
    def name(self):
        "Unique name that identifies the shared memory block."
        reported_name = self._name
        return reported_name

    @property
    def size(self):
        "Size in bytes."
        return self._size
