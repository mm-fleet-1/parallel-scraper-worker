"""CPU timing for the Grounding DINO tiny crop proposer (benchmark only; no model API calls, no writes).
Manifest = jsonl {"k": key, "u": blob URL without SAS}. Photos are fetched first (timed), then every variant runs on
every photo, interleaved per photo so variants share the same runner state. Prints one JSON summary line.
usage: python tools/det_bench.py --manifest-url URL [--threads 4]
env: MEDIA_BLOB_READ_SAS"""
import argparse, io, json, os, platform, re, resource, sys, time
import requests, torch
from PIL import Image

MODEL = "IDEA-Research/grounding-dino-tiny"
PROMPT = "a chest freezer. an ice cream freezer. a glass door refrigerator. a display counter. a shop signboard."
VARIANTS = {"800": (800, 1333, False), "600": (600, 1000, False), "480": (480, 800, False), "480_int8": (480, 800, True)}


def mask(e):
    return re.sub(r"sig=[^&\s]+", "sig=***", str(e))


def pct(xs, q):
    xs = sorted(xs); return round(xs[min(len(xs) - 1, int(len(xs) * q))], 3) if xs else None


def main():
    a = argparse.ArgumentParser(); a.add_argument("--manifest-url", required=True); a.add_argument("--threads", type=int, default=4)
    a = a.parse_args(); rs = os.environ["MEDIA_BLOB_READ_SAS"].lstrip("?"); s = requests.Session()
    try:
        r = s.get(f"{a.manifest_url}?{rs}", timeout=60); r.raise_for_status()
    except Exception as e:
        sys.exit(f"manifest read failed: {mask(e)}")
    rows = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    t0 = time.time(); ims, fetch = [], []
    for row in rows:
        t = time.time()
        try:
            b = s.get(f"{row['u'].split('?')[0]}?{rs}", timeout=60); b.raise_for_status()
            ims.append(Image.open(io.BytesIO(b.content)).convert("RGB")); fetch.append(time.time() - t)
        except Exception as e:
            print("fetch failed", row["k"], mask(e))
    if not ims:
        sys.exit("every photo failed")
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    t = time.time(); proc = AutoProcessor.from_pretrained(MODEL); model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).eval()
    load = time.time() - t
    q8 = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    T = {v: [] for v in VARIANTS}; B = {v: [] for v in VARIANTS}
    for i, im in enumerate(ims):
        for v, (se, le, int8) in VARIANTS.items():
            m = q8 if int8 else model; t = time.perf_counter()
            inp = proc(images=im, text=PROMPT, return_tensors="pt", size={"shortest_edge": se, "longest_edge": le})
            with torch.inference_mode():
                o = m(**inp)
            res = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=0.30, text_threshold=0.25,
                                                              target_sizes=[im.size[::-1]])[0]
            if i >= 2:  # first two photos = warm-up
                T[v].append(time.perf_counter() - t); B[v].append(len(res["scores"]))
    out = {"cpu": platform.processor() or open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "),
           "nproc": os.cpu_count(), "threads": a.threads, "photos": len(ims), "fetch_p50": pct(fetch, .5),
           "model_load_s": round(load, 1), "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
           "torch": torch.__version__, "total_s": round(time.time() - t0)}
    for v in VARIANTS:
        out[v] = {"p50": pct(T[v], .5), "p95": pct(T[v], .95), "mean": round(sum(T[v]) / len(T[v]), 3),
                  "photos_per_hour": round(3600 / (sum(T[v]) / len(T[v]))), "boxes": round(sum(B[v]) / len(B[v]), 1)}
    print("DETBENCH " + json.dumps(out))


if __name__ == "__main__":
    main()
