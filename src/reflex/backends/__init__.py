"""Backends that compute the readout somewhere other than this process."""

from __future__ import annotations


class BackendError(RuntimeError):
    """An upstream inference server failed us.

    Distinct from a bad request (422) and from our own bugs (500): the readout could not
    be computed because something the backend depends on refused, dropped the connection
    or timed out. `reflex.server` turns it into a 502 so a caller knows to retry."""
