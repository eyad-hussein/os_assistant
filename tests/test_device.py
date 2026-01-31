from unittest.mock import patch

from dagent.utils import get_os_name


def test_macos():
    with patch("platform.system", return_value="Darwin"):
        assert get_os_name() == "macOS"


def test_linux_ubuntu():
    with (
        patch("platform.system", return_value="Linux"),
        patch(
            "platform.freedesktop_os_release",
            return_value={"NAME": "Ubuntu"},
        ),
    ):
        assert get_os_name() == "Linux (Ubuntu)"


def test_linux_unknown():
    with (
        patch("platform.system", return_value="Linux"),
        patch(
            "platform.freedesktop_os_release",
            return_value={},
        ),
    ):
        assert get_os_name() == "Linux (Unknown)"


def test_windows():
    with patch("platform.system", return_value="Windows"):
        assert get_os_name() == "Windows"
