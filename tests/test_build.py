"""Test module for making sure model package is working"""

import os_assistant
from os_assistant import __version__


def test_build():
    """Make sure pytest is working."""
    assert True
    assert os_assistant
    assert __version__
