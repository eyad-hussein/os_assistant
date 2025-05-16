import base64
import pathlib
import zlib

import requests


def mermaid_to_png(mermaid_txt: str, out_path="linux_assistant_graph.png"):
    """
    Render Mermaid text → PNG via kroki.io (no local mermaid‑cli needed).
    """
    # 1) Kroki expects the diagram compressed + base64url encoded
    compressed = zlib.compress(mermaid_txt.encode("utf‑8"), level=9)
    b64url = base64.urlsafe_b64encode(compressed).decode("ascii")

    # 2) Call Kroki
    url = f"https://kroki.io/mermaid/png/{b64url}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # 3) Write to disk
    out = pathlib.Path(out_path)
    out.write_bytes(resp.content)
    return out
