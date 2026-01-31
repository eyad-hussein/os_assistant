import platform


def get_os_name():
    os_type = platform.system()
    if os_type == "Darwin":
        return "macOS"
    if os_type == "Linux":
        info = platform.freedesktop_os_release()
        os_type = f"Linux ({info.get('NAME', 'Unknown')})"
    return os_type
