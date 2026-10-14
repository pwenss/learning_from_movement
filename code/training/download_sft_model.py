"""Fetch official Qwen files from ModelScope and verify against a pinned HF manifest."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request


def digest(path, kind="sha256"):
    h = hashlib.new(kind)
    if kind == "sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    expected = json.loads(Path(args.manifest).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    url = "https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-VL-8B-Instruct/repo/files?Revision=master&Recursive=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        source = json.load(response)["Data"]
    files = {x["Path"]: x for x in source["Files"]}
    selected = [x for x in expected["files"] if x["rfilename"] != ".gitattributes"]
    for item in selected:
        ms = files[item["rfilename"]]
        assert ms["Size"] == item["size"], item["rfilename"]
        if "lfs" in item:
            assert ms["Sha256"] == item["lfs"]["sha256"], item["rfilename"]
    atomic(out / "source_manifest.json", {"huggingface": expected, "modelscope": source})
    progress, mu = {}, threading.Lock()

    def valid(path, item):
        if not path.exists() or path.stat().st_size != item["size"]:
            return False
        key = "sha256" if "lfs" in item else "sha1"
        match = item["lfs"]["sha256"] if "lfs" in item else item["blobId"]
        return digest(path, key) == match

    def fetch(item):
        name = item["rfilename"]
        final = out / name
        if valid(final, item):
            with mu:
                progress[name] = {"bytes": item["size"], "verified": True}
            return
        partial = out / (name + ".partial")
        ms = files[name]
        path = urllib.parse.quote(name)
        url = f"https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct/resolve/{ms['Revision']}/{path}"
        for attempt in range(1, 6):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if offset and response.status != 206:
                        offset = 0
                    with partial.open("ab" if offset else "wb") as f:
                        while True:
                            chunk = response.read(4 * 1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            with mu:
                                progress[name] = {"bytes": f.tell(), "verified": False, "attempt": attempt}
                assert valid(partial, item), f"File verification failed: {name}"
                os.replace(partial, final)
                with mu:
                    progress[name] = {"bytes": item["size"], "verified": True}
                print("VERIFIED", name, item["size"], flush=True)
                return
            except Exception as error:
                print("RETRY", name, attempt, type(error).__name__, str(error)[:160], flush=True)
                if partial.exists() and partial.stat().st_size >= item["size"]:
                    partial.unlink()
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 30))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, x) for x in selected]
        while not all(x.done() for x in futures):
            with mu:
                snapshot = dict(progress)
            atomic(out / "download_progress.json", {"time": time.time(), "files": snapshot})
            time.sleep(10)
        for future in futures:
            future.result()
    atomic(out / "model_verification.json", {"verified": True, "time": time.time(), "revision": expected["revision"], "files": progress})
    print("MODEL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
