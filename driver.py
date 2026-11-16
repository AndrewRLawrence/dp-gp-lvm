"""Run jobs in parallel subprocesses, the most expensive first."""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import patch  # noqa
from jobs import all_jobs

p = argparse.ArgumentParser()
p.add_argument("--script", default="phase_a.py")
p.add_argument("--workers", type=int, default=6)
p.add_argument("--filter", default="")
p.add_argument("--tag", default="baseline")
p.add_argument("--results", default=str(HERE / "results"))
p.add_argument("extra", nargs="*")
args = p.parse_args()
jobs = sorted((j for j in all_jobs() if args.filter in j.name), key=lambda j: -j.cost)
logs = HERE / "logs" / args.tag
logs.mkdir(parents=True, exist_ok=True)

def run(job):
    with open(logs / f"{job.name}.log", "w") as log:
        t = time.time()
        code = subprocess.call([sys.executable, str(HERE / args.script), "--job", job.name, "--tag", args.tag, "--results", args.results, *args.extra], stdout=log, stderr=subprocess.STDOUT)
    print(f"{time.strftime('%H:%M:%S')} {job.name} exit {code} ({time.time() - t:.0f} s)", flush=True)
    return code

with ThreadPoolExecutor(args.workers) as pool:
    codes = list(pool.map(run, jobs))
print("failed:", [j.name for j, c in zip(jobs, codes) if c != 0], flush=True)
