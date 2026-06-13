"""Board-scaling eval bench: 9x9 small net vs candidate 15x15 nets on MPS.

Evidence cell for wiki/topics/15x15-era-feasibility-and-plan.md. Measures pure
fp16 NN-eval throughput (no MCTS, no game loop) at the two production wave
sizes, to test whether the dispatch-bound regime makes a 15x15 board/bigger
net affordable on the M5 Max.

Run:  python scripts/bench_board_scaling.py        # GPU must be free (no derby)

Caveats (read before trusting): cold, eval-only, single-process microbench.
Per wiki/topics/perf-bench-vs-real-training-cost.md and the ingest-flooding
lesson, live training adds a contention tax (~30% at 9x9) and flood-scale
effects this bench cannot see. The go/no-go gate is a live run_sweep smoke
slice, not this number.
"""
import time

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(x + y)


class Net(nn.Module):
    """Mirrors gomoku/model.py's shape: stem -> res tower -> policy/value heads.

    Raw (unfused) BatchNorm in eval mode = slightly conservative vs the
    production Conv+BN-fused eval path.
    """

    def __init__(self, board, ch, blocks, in_planes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.pol = nn.Sequential(nn.Conv2d(ch, 2, 1), nn.Flatten(),
                                 nn.Linear(2 * board * board, board * board))
        self.val = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Flatten(),
                                 nn.Linear(board * board, 64), nn.ReLU(),
                                 nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        h = self.tower(self.stem(x))
        return self.pol(h), self.val(h)


def bench(board, ch, blocks, batch, secs=6.0):
    dev = torch.device("mps")
    net = Net(board, ch, blocks).to(dev).half().eval()
    x = torch.randn(batch, 4, board, board, device=dev, dtype=torch.half)
    with torch.no_grad():
        for _ in range(20):  # warmup
            net(x)
        torch.mps.synchronize()
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < secs:
            net(x)
            n += 1
        torch.mps.synchronize()
        dt = time.perf_counter() - t0
    params = sum(p.numel() for p in net.parameters())
    evals = n * batch / dt
    print(f"board={board:2d} ch={ch:3d} blocks={blocks:2d} params={params/1e3:7.0f}k "
          f"batch={batch:3d}: {evals:9.0f} evals/s  ({dt/n*1000:6.2f} ms/wave)")
    return evals


def main():
    print(f"torch {torch.__version__}, mps={torch.backends.mps.is_available()}")
    configs = [
        ("9x9 small (current champ arch)", 9, 64, 4),
        ("15x15 same-net (64x4)", 15, 64, 4),
        ("15x15 mid (96x8)", 15, 96, 8),
        ("15x15 big (128x10)", 15, 128, 10),
    ]
    results = {}
    for name, b, c, k in configs:
        print(f"-- {name}")
        for batch in (64, 512):
            results[(name, batch)] = bench(b, c, k, batch)

    base64 = results[("9x9 small (current champ arch)", 64)]
    base512 = results[("9x9 small (current champ arch)", 512)]
    print("\nRatios vs current 9x9 small:")
    for name, _, _, _ in configs[1:]:
        print(f"  {name:28s} batch=64: {base64/results[(name, 64)]:.2f}x slower | "
              f"batch=512: {base512/results[(name, 512)]:.2f}x slower")


if __name__ == "__main__":
    main()
