#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

/* Native lookahead (negamax + alpha-beta) search — issue #110.
 *
 * A move-identical port of gomoku/baselines.py's lookahead player:
 * `best_actions(board, depth, move_count)` returns the same tied-best root
 * action list, in the same order, as `baselines._root_best_actions` — same
 * stable move ordering, same alpha-beta cut decisions, same mate scoring,
 * same depth-0 forced-block quiescence. All heuristic weights are
 * integer-valued doubles, so score arithmetic is exact and comparisons match
 * the NumPy path bit-for-bit. Parity is enforced by
 * tests/test_lookahead_native.py.
 *
 * Board size is a COMPILE-TIME parameter, same shim pattern as
 * _state_ops_native.c (see _lookahead_native11.c etc. and setup.py). */

#ifndef BOARD_SIZE
#define BOARD_SIZE 9
#endif
#define N_CELLS (BOARD_SIZE * BOARD_SIZE)
#define WIN_LEN 5
#define MAX_BRANCH 12

#ifndef GOMOKU_LOOKAHEAD_MODULE_NAME
#define GOMOKU_LOOKAHEAD_MODULE_NAME _lookahead_native
#endif
#define GOMOKU_STR_(x) #x
#define GOMOKU_STR(x) GOMOKU_STR_(x)
#define GOMOKU_CAT_(a, b) a##b
#define GOMOKU_CAT(a, b) GOMOKU_CAT_(a, b)

/* Windows: one entry per length-5 line, dirs (0,1),(1,0),(1,1),(1,-1) in the
 * exact construction order of baselines._build_windows. */
#define MAX_WINDOWS (4 * N_CELLS)
static int16_t g_windows[MAX_WINDOWS][WIN_LEN];
static int g_n_windows = 0;
/* Per-cell list of windows through that cell (ascending window index). */
#define MAX_WIN_PER_CELL 32
static int16_t g_through[N_CELLS][MAX_WIN_PER_CELL];
static int8_t g_n_through[N_CELLS];
/* Chebyshev-2 neighborhood per cell (for candidate dilation). */
#define MAX_NEIGH 24
static int16_t g_neigh[N_CELLS][MAX_NEIGH];
static int8_t g_n_neigh[N_CELLS];

/* Weights — exact copies of baselines._MY_W / _OPP_W (already *1.5). */
static const double MY_W[6] = {0.0, 1.0, 25.0, 500.0, 10000.0, 10000000.0};
static const double OPP_W[6] = {0.0, 1.5, 60.0, 1125.0, 22500.0, 15000000.0};

static const double MATE = 1000000000.0;

static void build_tables(void) {
    static const int dirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    g_n_windows = 0;
    for (int d = 0; d < 4; d++) {
        int dr = dirs[d][0], dc = dirs[d][1];
        for (int r0 = 0; r0 < BOARD_SIZE; r0++) {
            for (int c0 = 0; c0 < BOARD_SIZE; c0++) {
                int r_end = r0 + dr * (WIN_LEN - 1);
                int c_end = c0 + dc * (WIN_LEN - 1);
                if (r_end < 0 || r_end >= BOARD_SIZE || c_end < 0 ||
                    c_end >= BOARD_SIZE) {
                    continue;
                }
                for (int k = 0; k < WIN_LEN; k++) {
                    g_windows[g_n_windows][k] =
                        (int16_t)((r0 + dr * k) * BOARD_SIZE + (c0 + dc * k));
                }
                g_n_windows++;
            }
        }
    }
    for (int c = 0; c < N_CELLS; c++) {
        g_n_through[c] = 0;
    }
    for (int w = 0; w < g_n_windows; w++) {
        for (int k = 0; k < WIN_LEN; k++) {
            int c = g_windows[w][k];
            g_through[c][g_n_through[c]++] = (int16_t)w;
        }
    }
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            int i = r * BOARD_SIZE + c;
            g_n_neigh[i] = 0;
            for (int dr = -2; dr <= 2; dr++) {
                for (int dc = -2; dc <= 2; dc++) {
                    if (dr == 0 && dc == 0) continue;
                    int rr = r + dr, cc = c + dc;
                    if (rr < 0 || rr >= BOARD_SIZE || cc < 0 || cc >= BOARD_SIZE)
                        continue;
                    g_neigh[i][g_n_neigh[i]++] =
                        (int16_t)(rr * BOARD_SIZE + cc);
                }
            }
        }
    }
}

/* ---- search state: two planes + incrementally-maintained window counts ---- */

typedef struct {
    uint8_t me[N_CELLS];
    uint8_t op[N_CELLS];
    int16_t cnt_me[MAX_WINDOWS]; /* my stones per window */
    int16_t cnt_op[MAX_WINDOWS];
} Pos;

/* Place a stone for the side owning (plane, cnt); returns 1 if it makes 5. */
static inline int place(uint8_t *plane, int16_t *cnt, int a) {
    int made5 = 0;
    plane[a] = 1;
    for (int j = 0; j < g_n_through[a]; j++) {
        int w = g_through[a][j];
        if (++cnt[w] >= WIN_LEN) made5 = 1;
    }
    return made5;
}

static inline void unplace(uint8_t *plane, int16_t *cnt, int a) {
    plane[a] = 0;
    for (int j = 0; j < g_n_through[a]; j++) {
        cnt[g_through[a][j]]--;
    }
}

/* Static eval from (cnt_me = side to move)'s perspective — exact mirror of
 * baselines.evaluate_position (window counts never exceed 5, no clip). */
static double eval_pos(const int16_t *cnt_me, const int16_t *cnt_op) {
    double s = 0.0;
    for (int w = 0; w < g_n_windows; w++) {
        if (cnt_op[w] == 0) s += MY_W[cnt_me[w]];
        if (cnt_me[w] == 0) s -= OPP_W[cnt_op[w]];
    }
    return s;
}

/* Per-cell placement score — exact mirror of baselines._score_cells. */
static inline double score_cell(const int16_t *cnt_me, const int16_t *cnt_op,
                                int c) {
    double offense = 0.0, defense = 0.0;
    for (int j = 0; j < g_n_through[c]; j++) {
        int w = g_through[c][j];
        int my = cnt_me[w], op = cnt_op[w];
        if (op == 0) {
            int m = my + 1;
            offense += MY_W[m > WIN_LEN ? WIN_LEN : m];
        }
        if (my == 0) {
            defense += OPP_W[op > WIN_LEN ? WIN_LEN : op];
        }
    }
    return offense + defense;
}

/* Does placing at empty cell c complete five for the side counted by cnt?
 * Mirror of baselines._find_immediate_wins' per-cell condition. */
static inline int completes_five(const int16_t *cnt, int c) {
    for (int j = 0; j < g_n_through[c]; j++) {
        if (cnt[g_through[c][j]] + 1 >= WIN_LEN) return 1;
    }
    return 0;
}

/* Candidate moves — mirror of baselines._candidate_moves: empty cells within
 * Chebyshev-2 of any stone, ascending; empty board -> center; crowded
 * pathology -> all legal. Returns count. */
static int candidate_moves(const Pos *p, int16_t *out) {
    uint8_t near[N_CELLS];
    int any_stone = 0;
    memset(near, 0, sizeof(near));
    for (int c = 0; c < N_CELLS; c++) {
        if (p->me[c] | p->op[c]) {
            any_stone = 1;
            for (int j = 0; j < g_n_neigh[c]; j++) near[g_neigh[c][j]] = 1;
        }
    }
    if (!any_stone) {
        out[0] = (int16_t)((BOARD_SIZE / 2) * BOARD_SIZE + BOARD_SIZE / 2);
        return 1;
    }
    int n = 0;
    for (int c = 0; c < N_CELLS; c++) {
        if (near[c] && !(p->me[c] | p->op[c])) out[n++] = (int16_t)c;
    }
    if (n == 0) {
        for (int c = 0; c < N_CELLS; c++) {
            if (!(p->me[c] | p->op[c])) out[n++] = (int16_t)c;
        }
    }
    return n;
}

/* Stable descending insertion sort of cand[] by score[] (ties keep ascending
 * cell order) — matches np.argsort(-scores, kind="stable"). */
static void sort_by_score_desc(int16_t *cand, double *score, int n) {
    for (int i = 1; i < n; i++) {
        int16_t c = cand[i];
        double s = score[i];
        int j = i - 1;
        while (j >= 0 && score[j] < s) {
            cand[j + 1] = cand[j];
            score[j + 1] = score[j];
            j--;
        }
        cand[j + 1] = c;
        score[j + 1] = s;
    }
}

/* Depth-0 forced-block quiescence — exact mirror of the depth==0 branch in
 * baselines._negamax. cnt_me/cnt_op are from side-to-move's perspective. */
static double quiescence(Pos *p, int16_t *cnt_me, int16_t *cnt_op,
                         uint8_t *me_plane) {
    int any = 0;
    double best_q = -INFINITY;
    for (int c = 0; c < N_CELLS; c++) {
        if (p->me[c] | p->op[c]) continue;       /* legal = empty */
        if (!completes_five(cnt_op, c)) continue; /* opp winning square? */
        any = 1;
        int made5 = place(me_plane, cnt_me, c); /* we block */
        if (made5) {
            unplace(me_plane, cnt_me, c);
            return MATE;
        }
        /* evaluate_position(child) is from the CHILD's side to move (our
         * opponent); negate to get back to ours. */
        double q = -eval_pos(cnt_op, cnt_me);
        unplace(me_plane, cnt_me, c);
        if (q > best_q) best_q = q;
    }
    if (!any) return eval_pos(cnt_me, cnt_op);
    return best_q;
}

/* Negamax with alpha-beta — exact mirror of baselines._negamax. The caller
 * has already ruled out "previous move made five" (it checks before
 * recursing, mirroring the Python in-loop terminal check). */
static double negamax(Pos *p, uint8_t *me, uint8_t *op, int16_t *cnt_me,
                      int16_t *cnt_op, int depth, double alpha, double beta,
                      int move_count) {
    if (move_count >= N_CELLS) return 0.0; /* draw */
    if (depth == 0) return quiescence(p, cnt_me, cnt_op, me);

    int16_t cand[N_CELLS];
    double score[N_CELLS];
    int n = candidate_moves(p, cand);
    for (int i = 0; i < n; i++) score[i] = score_cell(cnt_me, cnt_op, cand[i]);
    sort_by_score_desc(cand, score, n);
    if (n > MAX_BRANCH) n = MAX_BRANCH;

    double best = -INFINITY;
    for (int i = 0; i < n; i++) {
        int a = cand[i];
        int made5 = place(me, cnt_me, a);
        if (made5) {
            unplace(me, cnt_me, a);
            return MATE + (double)depth; /* we just won — shorter mates first */
        }
        double val;
        if (move_count + 1 >= N_CELLS) {
            val = 0.0; /* child is a terminal draw */
        } else {
            val = -negamax(p, op, me, cnt_op, cnt_me, depth - 1, -beta, -alpha,
                           move_count + 1);
        }
        unplace(me, cnt_me, a);
        if (val > best) best = val;
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
    }
    return best;
}

/* Root — exact mirror of baselines._root_best_actions. Writes the tied-best
 * action list into out[], returns its length. */
static int root_best_actions(Pos *p, int depth, int move_count, int16_t *out) {
    /* Step 1: immediate wins, tie-set by placement score. */
    int16_t wins[N_CELLS];
    int n_wins = 0;
    for (int c = 0; c < N_CELLS; c++) {
        if (p->me[c] | p->op[c]) continue;
        if (completes_five(p->cnt_me, c)) wins[n_wins++] = (int16_t)c;
    }
    if (n_wins > 0) {
        double best_s = -INFINITY;
        int n_out = 0;
        for (int i = 0; i < n_wins; i++) {
            double s = score_cell(p->cnt_me, p->cnt_op, wins[i]);
            if (s > best_s) {
                best_s = s;
                n_out = 0;
            }
            if (s == best_s) out[n_out++] = wins[i];
        }
        return n_out;
    }

    int16_t cand[N_CELLS];
    double score[N_CELLS];
    int n = candidate_moves(p, cand);
    for (int i = 0; i < n; i++)
        score[i] = score_cell(p->cnt_me, p->cnt_op, cand[i]);
    sort_by_score_desc(cand, score, n); /* NO branch cap at the root */

    double best_val = -INFINITY;
    double alpha = -INFINITY, beta = INFINITY;
    int n_out = 0;
    for (int i = 0; i < n; i++) {
        int a = cand[i];
        int made5 = place(p->me, p->cnt_me, a);
        double val;
        if (made5) {
            val = MATE + (double)depth; /* defensive; my_wins caught these */
        } else if (move_count + 1 >= N_CELLS) {
            val = 0.0; /* terminal draw child: -negamax entry draw = -0.0 */
        } else {
            val = -negamax(p, p->op, p->me, p->cnt_op, p->cnt_me, depth - 1,
                           -beta, -alpha, move_count + 1);
        }
        unplace(p->me, p->cnt_me, a);
        if (val > best_val) {
            best_val = val;
            n_out = 0;
            out[n_out++] = (int16_t)a;
        } else if (val == best_val) {
            out[n_out++] = (int16_t)a;
        }
        if (best_val > alpha) alpha = best_val;
    }
    return n_out;
}

/* ---- Python interface ---- */

static PyObject *py_best_actions(PyObject *self, PyObject *args) {
    PyObject *board_obj;
    int depth, move_count;
    if (!PyArg_ParseTuple(args, "Oii", &board_obj, &depth, &move_count)) {
        return NULL;
    }
    PyArrayObject *board = (PyArrayObject *)PyArray_FROM_OTF(
        board_obj, NPY_BOOL, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!board) return NULL;
    if (PyArray_NDIM(board) != 3 || PyArray_DIM(board, 0) != 2 ||
        PyArray_DIM(board, 1) != BOARD_SIZE ||
        PyArray_DIM(board, 2) != BOARD_SIZE) {
        Py_DECREF(board);
        PyErr_Format(PyExc_ValueError, "board must have shape (2, %d, %d)",
                     BOARD_SIZE, BOARD_SIZE);
        return NULL;
    }
    if (depth < 1) {
        Py_DECREF(board);
        PyErr_SetString(PyExc_ValueError, "depth must be >= 1");
        return NULL;
    }

    Pos pos;
    memset(&pos, 0, sizeof(pos));
    const npy_bool *data = (const npy_bool *)PyArray_DATA(board);
    for (int c = 0; c < N_CELLS; c++) {
        pos.me[c] = data[c] ? 1 : 0;
        pos.op[c] = data[N_CELLS + c] ? 1 : 0;
    }
    for (int w = 0; w < g_n_windows; w++) {
        int16_t m = 0, o = 0;
        for (int k = 0; k < WIN_LEN; k++) {
            m += pos.me[g_windows[w][k]];
            o += pos.op[g_windows[w][k]];
        }
        pos.cnt_me[w] = m;
        pos.cnt_op[w] = o;
    }

    int16_t out[N_CELLS];
    int n_out;
    Py_BEGIN_ALLOW_THREADS
    n_out = root_best_actions(&pos, depth, move_count, out);
    Py_END_ALLOW_THREADS
    Py_DECREF(board);

    PyObject *list = PyList_New(n_out);
    if (!list) return NULL;
    for (int i = 0; i < n_out; i++) {
        PyObject *v = PyLong_FromLong((long)out[i]);
        if (!v) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, v);
    }
    return list;
}

static PyMethodDef Methods[] = {
    {"best_actions", py_best_actions, METH_VARARGS,
     "best_actions(board, depth, move_count) -> list of tied-best root "
     "actions, move-identical to baselines._root_best_actions."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    GOMOKU_STR(GOMOKU_LOOKAHEAD_MODULE_NAME),
    "Native lookahead (negamax + alpha-beta) search, issue #110.",
    -1,
    Methods,
};

PyMODINIT_FUNC GOMOKU_CAT(PyInit_, GOMOKU_LOOKAHEAD_MODULE_NAME)(void) {
    import_array();
    build_tables();
    return PyModule_Create(&moduledef);
}
