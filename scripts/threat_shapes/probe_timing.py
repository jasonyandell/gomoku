"""Where does the time go? Isolate: import, first-call (compile), warm-call wall,
batch throughput, and node-budget sensitivity. One process."""
import os, time
os.environ.setdefault("GOMOKU_BOARD_SIZE", "15")
t = time.time()
import numpy as np
from gomoku.board_config import BOARD_SIZE as N
from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb
print(f"import:                 {time.time()-t:6.2f}s")

rng = np.random.default_rng(0)
def rand_boards(B, stones=12):
    bs = np.zeros((B, 2, N, N), bool)
    for b in range(B):
        idx = rng.choice(N*N, size=2*stones, replace=False)
        for k, i in enumerate(idx):
            bs[b, k % 2, i//N, i%N] = True
    return bs

one = rand_boards(1)
t = time.time(); solve_vct_mega_bb(one, max_nodes=2000); print(f"1st call B=1 mn=2000:   {time.time()-t:6.2f}s   (compile + run)")
t = time.time(); solve_vct_mega_bb(one, max_nodes=2000); print(f"warm  B=1 mn=2000:      {time.time()-t:6.2f}s   (pure wall)")
t = time.time(); solve_vct_mega_bb(one, max_nodes=20000); print(f"warm  B=1 mn=20000:     {time.time()-t:6.2f}s")
t = time.time(); solve_vct_mega_bb(one, max_nodes=200); print(f"warm  B=1 mn=200:       {time.time()-t:6.2f}s")

for B in (16, 256, 4096, 16384):
    bb = rand_boards(B)
    t = time.time(); solve_vct_mega_bb(bb, max_nodes=2000); dt = time.time()-t
    print(f"warm  B={B:<5} mn=2000:  {dt:6.2f}s   ({B/dt:8.0f} boards/s)")
