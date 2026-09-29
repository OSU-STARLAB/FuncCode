#!/usr/bin/env python
"""Two-GPU job dispatcher for the conv-KAGN CIFAR grid.

Why a queue and not DDP: the grid is many independent configs, several of which
are small. Running one whole config per GPU gives near-linear scaling with no
gradient-sync overhead and no risk that a single crash takes down both cards.
DDP is still available inside a single config via ``--ddp`` on the driver, and
is only worth it for the 8-layer CIFAR-100 dense stage.

State lives in ``<queue-dir>/status.json`` and is rewritten after every
transition, so an external monitor can poll it without touching the runner. Jobs
are keyed by name; a job whose ``done`` marker exists is skipped on restart, so
the queue is safely resumable after a crash, a preemption, or a Ctrl-C.

Usage
-----
  python tools/gpu_queue.py --plan scripts/paper/plans/cifar_tier1.json --gpus 0 1
  python tools/gpu_queue.py --plan ... --status-only     # print status and exit
  python tools/gpu_queue.py --plan ... --dry-run         # show the commands
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Job:
    name: str
    cmd: List[str]
    est_minutes: int = 60
    priority: int = 100          # lower runs first
    depends_on: Optional[str] = None
    state: str = "pending"       # pending running done failed skipped
    gpu: Optional[int] = None
    attempts: int = 0
    started: Optional[float] = None
    finished: Optional[float] = None
    returncode: Optional[int] = None
    log: Optional[str] = None


class Queue:
    def __init__(self, plan_path: Path, queue_dir: Path, gpus: List[int],
                 max_retries: int = 1, poll: float = 10.0, slots_per_gpu: int = 1):
        self.plan_path = plan_path
        self.dir = queue_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        (self.dir / "done").mkdir(exist_ok=True)
        self.gpus = gpus
        # A slot is one concurrent job. These nets are small (the 8-layer
        # CIFAR-100 model peaks at ~1.2 GiB), so the GPU is latency- rather than
        # memory-bound: two jobs per card measured 1.35x the throughput of one,
        # three measured no better than two. Slots round-robin over the cards so
        # that the first len(gpus) jobs still land one per card.
        self.slots: List[int] = list(range(len(gpus) * max(1, slots_per_gpu)))
        self.slot_gpu: Dict[int, int] = {s: gpus[s % len(gpus)] for s in self.slots}
        self.max_retries = max_retries
        self.poll = poll
        self.jobs: Dict[str, Job] = {}
        self.running: Dict[int, tuple] = {}   # slot -> (Job, Popen, filehandle)
        self._load_plan()
        self._restore()

    # -- plan / state ----------------------------------------------------
    def _load_plan(self):
        spec = json.loads(self.plan_path.read_text())
        for j in spec["jobs"]:
            cmd = j["cmd"]
            if isinstance(cmd, str):
                cmd = shlex.split(cmd)
            self.jobs[j["name"]] = Job(
                name=j["name"], cmd=cmd, est_minutes=j.get("est_minutes", 60),
                priority=j.get("priority", 100), depends_on=j.get("depends_on"))

    def _restore(self):
        for job in self.jobs.values():
            if (self.dir / "done" / f"{job.name}.ok").exists():
                job.state = "done"

    def _write_status(self):
        now = time.time()
        jobs = [asdict(j) for j in self.jobs.values()]
        counts: Dict[str, int] = {}
        for j in self.jobs.values():
            counts[j.state] = counts.get(j.state, 0) + 1

        remaining = sum(j.est_minutes for j in self.jobs.values() if j.state in ("pending", "running"))
        eta_min = remaining / max(len(self.slots), 1)

        payload = {
            "updated": now,
            "updated_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "plan": str(self.plan_path),
            "gpus": self.gpus,
            "slots": len(self.slots),
            "counts": counts,
            "total": len(self.jobs),
            "eta_minutes_optimistic": round(eta_min, 1),
            "running": {str(s): {"job": j.name, "gpu": self.slot_gpu[s],
                                 "elapsed_min": round((now - j.started) / 60, 1),
                                 "est_min": j.est_minutes, "log": j.log}
                        for s, (j, _, _) in self.running.items()},
            "jobs": jobs,
        }
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.dir / "status.json")

    # -- scheduling ------------------------------------------------------
    def _ready(self) -> List[Job]:
        out = []
        for j in self.jobs.values():
            if j.state != "pending":
                continue
            if j.depends_on:
                dep = self.jobs.get(j.depends_on)
                if dep is None or dep.state != "done":
                    continue
            out.append(j)
        # longest-first within a priority tier keeps the tail short
        return sorted(out, key=lambda j: (j.priority, -j.est_minutes))

    def _launch(self, job: Job, slot: int):
        gpu = self.slot_gpu[slot]
        log_path = self.dir / "logs" / f"{job.name}.log"
        fh = open(log_path, "a", buffering=1)
        fh.write(f"\n{'='*70}\n=== attempt {job.attempts+1} on GPU {gpu} (slot {slot}) at "
                 f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n=== {' '.join(job.cmd)}\n{'='*70}\n")

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Concurrent jobs on one node should not each grab every core.
        env["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 8) // max(len(self.slots), 1)))

        p = subprocess.Popen(job.cmd, stdout=fh, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        job.state, job.gpu, job.started = "running", gpu, time.time()
        job.attempts += 1
        job.log = str(log_path)
        self.running[slot] = (job, p, fh)
        print(f"[queue] GPU{gpu}/slot{slot} <- {job.name}  (~{job.est_minutes} min)", flush=True)

    def _reap(self):
        for slot in list(self.running):
            job, p, fh = self.running[slot]
            rc = p.poll()
            if rc is None:
                continue
            fh.close()
            del self.running[slot]
            job.finished, job.returncode = time.time(), rc
            mins = (job.finished - job.started) / 60
            where = f"GPU{self.slot_gpu[slot]}/slot{slot}"

            if rc == 0:
                job.state = "done"
                (self.dir / "done" / f"{job.name}.ok").write_text(
                    json.dumps({"minutes": mins, "finished": job.finished}))
                print(f"[queue] {where} -> {job.name} DONE in {mins:.1f} min", flush=True)
            elif job.attempts <= self.max_retries:
                job.state = "pending"
                print(f"[queue] {where} -> {job.name} FAILED rc={rc} after {mins:.1f} min; "
                      f"requeued ({job.attempts}/{self.max_retries+1})", flush=True)
            else:
                job.state = "failed"
                print(f"[queue] {where} -> {job.name} FAILED rc={rc}; giving up. "
                      f"Log: {job.log}", flush=True)

    def run(self):
        stop = {"flag": False}

        def handler(signum, frame):
            stop["flag"] = True
            print("\n[queue] shutdown requested; terminating children...", flush=True)
            for slot, (job, p, fh) in list(self.running.items()):
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass
            self._write_status()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        t0 = time.time()
        self._write_status()
        while not stop["flag"]:
            self._reap()
            free = [s for s in self.slots if s not in self.running]
            for job in self._ready():
                if not free:
                    break
                self._launch(job, free.pop(0))
            self._write_status()

            if not self.running and not self._ready():
                break
            time.sleep(self.poll)

        self._reap()
        self._write_status()
        done = sum(1 for j in self.jobs.values() if j.state == "done")
        failed = [j.name for j in self.jobs.values() if j.state == "failed"]
        print(f"\n[queue] finished in {(time.time()-t0)/60:.1f} min | "
              f"{done}/{len(self.jobs)} done", flush=True)
        if failed:
            print(f"[queue] FAILED: {', '.join(failed)}", flush=True)
        return 1 if failed else 0


def print_status(queue_dir: Path):
    path = queue_dir / "status.json"
    if not path.exists():
        print(f"no status at {path}")
        return 1
    s = json.loads(path.read_text())
    age = time.time() - s["updated"]
    print(f"updated {s['updated_human']} ({age/60:.1f} min ago)")
    print(f"counts: {s['counts']}   of {s['total']}   "
          f"optimistic ETA {s['eta_minutes_optimistic']/60:.1f} h"
          f"   ({s.get('slots', len(s['gpus']))} slots on GPUs {s['gpus']})")
    for slot, r in s.get("running", {}).items():
        print(f"  GPU{r.get('gpu', '?')}/slot{slot}: {r['job']}  "
              f"{r['elapsed_min']:.0f}/{r['est_min']} min")
    bad = [j for j in s["jobs"] if j["state"] == "failed"]
    if bad:
        print("FAILED:")
        for j in bad:
            print(f"  {j['name']}  rc={j['returncode']}  log={j['log']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--queue-dir", default="runs_conv_cifar/_queue")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--slots-per-gpu", type=int, default=1,
                    help="concurrent jobs per GPU; 2 measured 1.35x throughput "
                         "measured for these models; 3 gave no further gain")
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--status-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset-failed", action="store_true",
                    help="clear failed markers so they are retried")
    args = ap.parse_args()

    qdir = Path(args.queue_dir)
    if args.status_only:
        return print_status(qdir)

    q = Queue(Path(args.plan), qdir, args.gpus, args.max_retries, args.poll,
              args.slots_per_gpu)

    if args.reset_failed:
        for j in q.jobs.values():
            if j.state == "failed":
                j.state = "pending"
                j.attempts = 0

    if args.dry_run:
        for j in sorted(q.jobs.values(), key=lambda x: (x.priority, -x.est_minutes)):
            print(f"[{j.state:8s}] {j.name:44s} ~{j.est_minutes:4d}m  {' '.join(j.cmd)}")
        total = sum(j.est_minutes for j in q.jobs.values() if j.state != "done")
        print(f"\n{len(q.jobs)} jobs, {total} GPU-minutes, "
              f"~{total/len(q.slots)/60:.1f} h wall on {len(q.slots)} slots "
              f"({len(args.gpus)} GPUs x {args.slots_per_gpu})")
        return 0

    return q.run()


if __name__ == "__main__":
    sys.exit(main())
