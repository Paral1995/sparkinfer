#!/usr/bin/env python3
"""Automatic evaluation on a vast.ai GPU: provision (or reuse) → build/correctness/speed → label → teardown.

Requires VAST_API_KEY (`vastai set api-key <key>` or `export VAST_API_KEY=...`).
The numeric label is computed on-box by bench/scripts/label.py — this script only orchestrates.

  # reuse a running box (fast; gets the *current* ssh-url via the API, so port rotation is fine):
  python eval/vast_eval.py --reuse 42134865 --keep --frontier 164 --ceiling 366 --ref main

  # provision a fresh RTX 5090, evaluate, and destroy:
  python eval/vast_eval.py --ref <branch-or-commit> --frontier 164 --ceiling 366

Frontier/ceiling are the current best tok/s and the roofline (or reference, e.g. llama.cpp) cap.
"""
import argparse, json, os, subprocess, sys, time
from vastai import VastAI

REPO  = os.environ.get("EVAL_REPO",  "https://github.com/gittensor-ai-lab/sparkinfer")
IMAGE = os.environ.get("EVAL_IMAGE", "nvidia/cuda:12.8.0-devel-ubuntu24.04")  # needs nvcc for sm_120

def sh(host, port, cmd, timeout=3600):
    return subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
         "-p", str(port), f"root@{host}", cmd],
        capture_output=True, text=True, timeout=timeout)

def parse_ssh_url(url):                      # "ssh://root@HOST:PORT"
    u = url.replace("ssh://", "")
    return u.split("@")[-1].split(":")[0], int(u.rsplit(":", 1)[1])

def wait_ssh(host, port, tries=60):
    for _ in range(tries):
        try:
            if sh(host, port, "echo ok", timeout=15).stdout.strip().endswith("ok"): return True
        except Exception: pass
        time.sleep(10)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="main")
    ap.add_argument("--frontier", type=float, default=0)
    ap.add_argument("--ceiling",  type=float, default=0)
    ap.add_argument("--reuse", type=int, default=0, help="evaluate on an existing instance id")
    ap.add_argument("--keep", action="store_true", help="don't destroy after (implied for --reuse)")
    ap.add_argument("--gpu", default="RTX_5090")
    ap.add_argument("--image", default=IMAGE)
    args = ap.parse_args()

    v = VastAI()                              # reads VAST_API_KEY
    created = False
    iid = args.reuse
    if not iid:
        offers = v.search_offers(query=f"gpu_name={args.gpu} num_gpus=1 cuda_vers>=12.8 inet_down>=100",
                                 order="dph_total", limit=10)
        if not offers: sys.exit("no matching offers")
        off = offers[0]
        print(f">> creating instance: offer {off['id']} {off.get('gpu_name')} ${off.get('dph_total'):.3f}/hr")
        inst = v.create_instance(id=off["id"], image=args.image, disk=120, ssh=True, direct=True)
        iid = inst.get("new_contract") or inst.get("id"); created = True

    # wait until running, then resolve the *current* ssh endpoint via the API
    for _ in range(60):
        info = next((i for i in v.show_instances() if i.get("id") == iid), None)
        if info and info.get("actual_status") == "running" and info.get("ssh_host"): break
        time.sleep(10)
    host, port = parse_ssh_url(v.ssh_url(iid))
    print(f">> instance {iid}: ssh root@{host}:{port}")
    if not wait_ssh(host, port): sys.exit("ssh never came up")

    try:
        setup = ("export DEBIAN_FRONTEND=noninteractive; "
                 "command -v git >/dev/null || (apt-get update -q && apt-get install -y -q git curl cmake build-essential); "
                 "pip install -q --break-system-packages huggingface_hub tokenizers >/dev/null 2>&1 || true; "
                 f"rm -rf /root/sparkinfer && git clone -q {REPO} /root/sparkinfer")
        if sh(host, port, setup, timeout=1800).returncode: sys.exit("setup failed")

        ev = (f"cd /root/sparkinfer && MODELS_DIR=/workspace/models LLAMACPP_DIR=/workspace/.llamacpp "
              f"bench/scripts/evaluate.sh --ref {args.ref} --frontier {args.frontier} --ceiling {args.ceiling}")
        r = sh(host, port, ev, timeout=10800)
        sys.stdout.write(r.stdout[-4000:])
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON")), None)
        if line:
            print("\n=== VERDICT ===")
            print(json.dumps(json.loads(line[len("RESULT_JSON "):]), indent=2))
        else:
            print("\n!! no RESULT_JSON — stderr tail:\n" + r.stderr[-1500:])
    finally:
        if created and not args.keep:
            print(f">> destroying instance {iid}"); v.destroy_instance(id=iid)
        else:
            print(f">> leaving instance {iid} running (--keep / --reuse)")

if __name__ == "__main__":
    main()
