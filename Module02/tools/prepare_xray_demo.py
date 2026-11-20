"""Download the pinned image examples and pretrained weights before class.

Only preparation uses the network. Expected SHA-256 hashes live in the manifest.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "models/xray/manifest.json").read_text())
    for item in [*manifest["images"], manifest["weights"]]:
        path = ROOT / item["path"]
        expected = item.get("sha256")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(item["url"], headers={"User-Agent": "CPSC4970-demo"})
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
            if expected and hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Download checksum mismatch: {path.name}")
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_bytes(content)
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and digest != expected:
            raise ValueError(f"Cached file checksum mismatch: {path.name}")
        print(path.name, digest)


if __name__ == "__main__":
    main()
