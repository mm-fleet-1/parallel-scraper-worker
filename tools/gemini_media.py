"""One Gemini call over one media file held in the private media blob.

Generic on purpose: this repo is public, so no client content lives here. The media and the
prompt are read from blob, the answer is written back to blob, and only token counts and
cost reach the log.

  python tools/gemini_media.py --media <blob path> --prompt <blob path> --out <blob prefix>
         [--model gemini-3.8-flash] [--cap-usd 1.0] [--price-in 0.75] [--price-out 3.75] [--max-out 24000]

env: LLM_BATCH_GEMINI_KEY, MEDIA_BLOB_READ_SAS, MEDIA_BLOB_WRITE_SAS (query strings)
"""
import argparse
import json
import mimetypes
import os
import sys
import tempfile
import time

import requests
from google import genai
from google.genai import types

BASE = "https://micromarket.blob.core.windows.net/scraper-media"


def sas(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(f"{name} missing")
    return v if v.startswith("?") else "?" + v.split("?", 1)[-1]


def blob_get(path: str, dest: str) -> None:
    with requests.get(f"{BASE}/{path}{sas('MEDIA_BLOB_READ_SAS')}", stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def blob_put(path: str, data: bytes, ctype: str) -> None:
    r = requests.put(f"{BASE}/{path}{sas('MEDIA_BLOB_WRITE_SAS')}", data=data, timeout=300,
                     headers={"x-ms-blob-type": "BlockBlob", "Content-Type": ctype})
    r.raise_for_status()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--cap-usd", type=float, default=1.0)
    ap.add_argument("--price-in", type=float, default=0.75)
    ap.add_argument("--price-out", type=float, default=3.75)
    ap.add_argument("--max-out", type=int, default=24000)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp()
    media = os.path.join(tmp, os.path.basename(a.media))
    blob_get(a.media, media)
    prompt_file = os.path.join(tmp, "prompt.txt")
    blob_get(a.prompt, prompt_file)
    prompt = open(prompt_file, encoding="utf-8").read()
    mime = mimetypes.guess_type(media)[0] or "application/octet-stream"
    print(f"media {os.path.getsize(media) / 1e6:.0f} MB {mime}", flush=True)

    client = genai.Client(api_key=os.environ["LLM_BATCH_GEMINI_KEY"])
    f = client.files.upload(file=media, config={"mime_type": mime})
    while f.state.name == "PROCESSING":
        time.sleep(5)
        f = client.files.get(name=f.name)
    if f.state.name != "ACTIVE":
        sys.exit(f"file state {f.state.name}")
    try:
        contents = [types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=mime)), prompt]
        n = client.models.count_tokens(model=a.model, contents=contents).total_tokens
        worst = n / 1e6 * a.price_in + a.max_out / 1e6 * a.price_out
        print(f"input tokens {n:,}; worst case ${worst:.3f} (cap ${a.cap_usd})", flush=True)
        if worst >= a.cap_usd:
            sys.exit("over the cap: not run")
        r = client.models.generate_content(
            model=a.model, contents=contents,
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=a.max_out))
        u = r.usage_metadata
        out_tok = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
        usage = {"model": a.model, "prompt_tokens": u.prompt_token_count, "output_tokens": u.candidates_token_count,
                 "thinking_tokens": u.thoughts_token_count, "finish": str(r.candidates[0].finish_reason),
                 "usd": round((u.prompt_token_count or 0) / 1e6 * a.price_in + out_tok / 1e6 * a.price_out, 4)}
        blob_put(f"{a.out}/answer.md", (r.text or "").encode("utf-8"), "text/markdown; charset=utf-8")
        blob_put(f"{a.out}/usage.json", json.dumps(usage, indent=1).encode(), "application/json")
        print("usage", json.dumps(usage), flush=True)
    finally:
        client.files.delete(name=f.name)


if __name__ == "__main__":
    main()
