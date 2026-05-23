"""Synthetic GPU load generator for engine-isolation stress tests.

Runs a continuous large-matmul hot loop on the MPS (Apple GPU) device to
saturate the GPU, so the perf lab can measure whether a concurrent
CPU/ANE self-play worker pool is genuinely isolated from GPU pressure
(or secretly coupled via shared memory bandwidth / hidden GPU use).

Usage:
    python scripts/gpu_load_generator.py --secs 120 --matrix 8192

Prints a periodic effective TFLOP/s estimate so the operator can confirm
the GPU is actually under load. Kill with SIGINT/SIGTERM (clean exit).
See wiki/topics/coreml-design-envelope-and-our-fit.md (engine-isolation
stress test) and the L09g' lane in wiki/ops/perf-queue.md.
"""

import argparse
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--secs", type=float, default=120.0, help="run duration in seconds")
    ap.add_argument("--matrix", type=int, default=8192, help="square matrix dim for matmul")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--report-every", type=float, default=10.0, help="seconds between throughput prints")
    args = ap.parse_args()

    dev = torch.device(args.device)
    n = args.matrix
    a = torch.randn(n, n, device=dev, dtype=torch.float32)
    b = torch.randn(n, n, device=dev, dtype=torch.float32)
    # one matmul = 2*n^3 FLOPs
    flops_per_matmul = 2.0 * (n ** 3)

    print(f"[gpu-load] device={dev} matrix={n}x{n} secs={args.secs} "
          f"(~{flops_per_matmul/1e9:.1f} GFLOP/matmul)", flush=True)

    t_start = time.time()
    t_end = t_start + args.secs
    t_report = t_start + args.report_every
    count = 0
    count_at_report = 0
    while time.time() < t_end:
        c = a @ b
        # rotate so the result feeds forward (prevents the compiler from
        # eliding work and keeps the GPU continuously busy)
        a = c
        torch.mps.synchronize()
        count += 1
        now = time.time()
        if now >= t_report:
            window = now - (t_report - args.report_every)
            dn = count - count_at_report
            tflops = (dn * flops_per_matmul) / window / 1e12
            print(f"[gpu-load] t={now-t_start:5.1f}s matmuls={count} "
                  f"effective={tflops:.2f} TFLOP/s", flush=True)
            t_report = now + args.report_every
            count_at_report = count
            # reseed occasionally so values don't blow up to inf/nan and
            # turn the matmul into a degenerate (cheap) op
            a = torch.randn(n, n, device=dev, dtype=torch.float32)

    print(f"[gpu-load] done: {count} matmuls in {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
