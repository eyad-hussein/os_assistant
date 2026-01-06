import platform


def get_os_name():
    os_type = platform.system()
    if os_type == "Darwin":
        return "macOS"
    return os_type
