#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define BOARD_SIZE 9
#define N_ACTIONS 81
#define HISTORY_PLY 8
#define HISTORY_STORED 8
#define N_INPUT_PLANES 17
#define MAX_PATH N_ACTIONS
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    uint64_t lo;
    uint64_t hi;
} Bits;

typedef struct {
    Bits current;
    Bits opponent;
    Bits hist_current[HISTORY_STORED];
    Bits hist_opponent[HISTORY_STORED];
    int hist_len;
    int move_count;
} CState;

typedef struct {
    CState state;
    int parent;
    int parent_action;
    int child[N_ACTIONS];
    int32_t N[N_ACTIONS];
    float W[N_ACTIONS];
    float P[N_ACTIONS];
    float logits[N_ACTIONS];  // Gumbel: centered raw logits (illegal -> -INF).
    unsigned char legal[N_ACTIONS];
    unsigned char expanded;
    unsigned char is_terminal;
    float terminal_value;
} CNode;

typedef struct {
    PyObject_HEAD
    CNode *nodes;
    int node_count;
    int node_cap;
    int root;
    double c_puct;
    double c_puct_base;
    double dirichlet_alpha;
    double dirichlet_eps;
    // KataGo forced-playouts (Wu 2019). 0.0 == OFF == byte-identical legacy
    // behavior. When > 0, root-only selection forces each already-visited
    // root child up to n_forced(a) = ceil(sqrt(k * P(a) * N_root)) visits,
    // and policy() subtracts those forced visits back out (policy-target
    // pruning) before forming the training target.
    double forced_playout_k;
    uint64_t rng;
    // --- Gumbel AlphaZero root selection + Sequential Halving state. ---
    // gumbel_root==0 => ordinary PUCT (byte-identical legacy). When on, the
    // root edge is forced to a Sequential-Halving-scheduled candidate rather
    // than PUCT; internal nodes are unchanged PUCT.
    int gumbel_root;
    int gumbel_m;
    double gumbel_c_visit;
    double gumbel_c_scale;
    // Per-current-root Gumbel search state (rebuilt at each advance_root /
    // each native_gumbel_search_batch root setup).
    int gumbel_candidates[N_ACTIONS];   // survivor action ids (current phase)
    int gumbel_n_candidates;            // count of survivors
    double gumbel_noise[N_ACTIONS];     // g(a), drawn once per root search
    float root_value;                   // network value at the current root
    int gumbel_forced_action;           // the candidate to force on next sim (<0 = none)
    int gumbel_selected_action;         // SH-chosen final action (set by scheduler)
    unsigned char gumbel_sampled;       // 1 once topk has been drawn this root
} NativeMCTSGameObject;

typedef struct {
    int game_index;
    int leaf_index;
    int path_nodes[MAX_PATH];
    int path_actions[MAX_PATH];
    int path_len;
} Pending;

typedef struct {
    Pending *items;
    int count;
    int cap;
} PendingVec;

static PyTypeObject NativeMCTSGameType;

static Bits win_masks[160];
static int win_mask_count = 0;

static inline Bits bits_zero(void) {
    Bits b;
    b.lo = 0;
    b.hi = 0;
    return b;
}

static inline int bits_get(Bits b, int action) {
    if (action < 64) {
        return (int)((b.lo >> action) & 1ULL);
    }
    return (int)((b.hi >> (action - 64)) & 1ULL);
}

static inline void bits_set(Bits *b, int action) {
    if (action < 64) {
        b->lo |= (1ULL << action);
    } else {
        b->hi |= (1ULL << (action - 64));
    }
}

static inline Bits bits_or(Bits a, Bits b) {
    Bits out;
    out.lo = a.lo | b.lo;
    out.hi = a.hi | b.hi;
    return out;
}

static inline int bits_contains_all(Bits value, Bits mask) {
    return ((value.lo & mask.lo) == mask.lo) && ((value.hi & mask.hi) == mask.hi);
}

static void add_win_mask(int r, int c, int dr, int dc) {
    Bits mask = bits_zero();
    for (int k = 0; k < 5; k++) {
        int rr = r + dr * k;
        int cc = c + dc * k;
        bits_set(&mask, rr * BOARD_SIZE + cc);
    }
    win_masks[win_mask_count++] = mask;
}

static void init_win_masks(void) {
    if (win_mask_count != 0) {
        return;
    }
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c <= BOARD_SIZE - 5; c++) {
            add_win_mask(r, c, 0, 1);
        }
    }
    for (int r = 0; r <= BOARD_SIZE - 5; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            add_win_mask(r, c, 1, 0);
        }
    }
    for (int r = 0; r <= BOARD_SIZE - 5; r++) {
        for (int c = 0; c <= BOARD_SIZE - 5; c++) {
            add_win_mask(r, c, 1, 1);
        }
    }
    for (int r = 0; r <= BOARD_SIZE - 5; r++) {
        for (int c = 4; c < BOARD_SIZE; c++) {
            add_win_mask(r, c, 1, -1);
        }
    }
}

static int has_five_bits(Bits stones) {
    for (int i = 0; i < win_mask_count; i++) {
        if (bits_contains_all(stones, win_masks[i])) {
            return 1;
        }
    }
    return 0;
}

static uint64_t splitmix64_next(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double rng_uniform(NativeMCTSGameObject *game) {
    uint64_t x = splitmix64_next(&game->rng);
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}

static double rng_normal(NativeMCTSGameObject *game) {
    double u1 = rng_uniform(game);
    double u2 = rng_uniform(game);
    if (u1 < 1e-300) {
        u1 = 1e-300;
    }
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static double rng_gumbel(NativeMCTSGameObject *game) {
    // Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1). Clamp U away from 0 and 1
    // (matching gumbel.py's u = clip(u, 1e-300, 1.0); the inner -log(u) also
    // needs u < 1 to stay finite).
    double u = rng_uniform(game);
    if (u < 1e-300) {
        u = 1e-300;
    }
    if (u >= 1.0) {
        u = 1.0 - 1e-16;
    }
    return -log(-log(u));
}

static double rng_gamma(NativeMCTSGameObject *game, double alpha) {
    if (alpha <= 0.0) {
        return 0.0;
    }
    if (alpha < 1.0) {
        double u = rng_uniform(game);
        if (u < 1e-300) {
            u = 1e-300;
        }
        return rng_gamma(game, alpha + 1.0) * pow(u, 1.0 / alpha);
    }

    double d = alpha - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x = rng_normal(game);
        double v = 1.0 + c * x;
        if (v <= 0.0) {
            continue;
        }
        v = v * v * v;
        double u = rng_uniform(game);
        if (u < 1.0 - 0.0331 * x * x * x * x) {
            return d * v;
        }
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) {
            return d * v;
        }
    }
}

static void state_initial(CState *state) {
    memset(state, 0, sizeof(CState));
}

static int check_board_shape(PyArrayObject *arr) {
    if (PyArray_NDIM(arr) != 3) {
        PyErr_SetString(PyExc_ValueError, "board must have shape (2, 9, 9)");
        return 0;
    }
    npy_intp const *dims = PyArray_DIMS(arr);
    if (dims[0] != 2 || dims[1] != BOARD_SIZE || dims[2] != BOARD_SIZE) {
        PyErr_SetString(PyExc_ValueError, "board must have shape (2, 9, 9)");
        return 0;
    }
    return 1;
}

static int bits_from_board_array(PyArrayObject *arr, Bits *current, Bits *opponent) {
    if (!check_board_shape(arr)) {
        return 0;
    }
    npy_bool const *data = (npy_bool const *)PyArray_DATA(arr);
    *current = bits_zero();
    *opponent = bits_zero();
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            int a = r * BOARD_SIZE + c;
            if (data[a]) {
                bits_set(current, a);
            }
            if (data[N_ACTIONS + a]) {
                bits_set(opponent, a);
            }
        }
    }
    return 1;
}

static int state_from_py(PyObject *state_obj, CState *out) {
    state_initial(out);

    PyObject *board_obj = PyObject_GetAttrString(state_obj, "board");
    if (board_obj == NULL) {
        return 0;
    }
    PyArrayObject *board = (PyArrayObject *)PyArray_FROM_OTF(
        board_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    Py_DECREF(board_obj);
    if (board == NULL) {
        return 0;
    }
    int ok = bits_from_board_array(board, &out->current, &out->opponent);
    Py_DECREF(board);
    if (!ok) {
        return 0;
    }

    PyObject *move_obj = PyObject_GetAttrString(state_obj, "move_count");
    if (move_obj == NULL) {
        return 0;
    }
    long move_count = PyLong_AsLong(move_obj);
    Py_DECREF(move_obj);
    if (move_count == -1 && PyErr_Occurred()) {
        return 0;
    }
    out->move_count = (int)move_count;

    PyObject *history_obj = PyObject_GetAttrString(state_obj, "history");
    if (history_obj == NULL) {
        return 0;
    }
    PyObject *history = PySequence_Fast(history_obj, "history must be a sequence");
    Py_DECREF(history_obj);
    if (history == NULL) {
        return 0;
    }
    Py_ssize_t hist_len = PySequence_Fast_GET_SIZE(history);
    if (hist_len > HISTORY_STORED) {
        hist_len = HISTORY_STORED;
    }
    out->hist_len = (int)hist_len;
    PyObject **items = PySequence_Fast_ITEMS(history);
    for (Py_ssize_t i = 0; i < hist_len; i++) {
        PyArrayObject *hist_board = (PyArrayObject *)PyArray_FROM_OTF(
            items[i], NPY_BOOL, NPY_ARRAY_IN_ARRAY);
        if (hist_board == NULL) {
            Py_DECREF(history);
            return 0;
        }
        ok = bits_from_board_array(
            hist_board, &out->hist_current[i], &out->hist_opponent[i]);
        Py_DECREF(hist_board);
        if (!ok) {
            Py_DECREF(history);
            return 0;
        }
    }
    Py_DECREF(history);
    return 1;
}

static void fill_bits_plane(Bits bits, float *plane) {
    for (int a = 0; a < N_ACTIONS; a++) {
        if (bits_get(bits, a)) {
            plane[a] = 1.0f;
        }
    }
}

static void state_to_planes(CState const *state, float *out) {
    memset(out, 0, N_INPUT_PLANES * N_ACTIONS * sizeof(float));
    fill_bits_plane(state->current, out);
    fill_bits_plane(state->opponent, out + HISTORY_PLY * N_ACTIONS);
    for (int k = 1; k < HISTORY_PLY && k <= state->hist_len; k++) {
        Bits hist_current = state->hist_current[k - 1];
        Bits hist_opponent = state->hist_opponent[k - 1];
        if ((k % 2) == 0) {
            fill_bits_plane(hist_current, out + k * N_ACTIONS);
            fill_bits_plane(hist_opponent, out + (HISTORY_PLY + k) * N_ACTIONS);
        } else {
            fill_bits_plane(hist_opponent, out + k * N_ACTIONS);
            fill_bits_plane(hist_current, out + (HISTORY_PLY + k) * N_ACTIONS);
        }
    }
    float *const_plane = out + (2 * HISTORY_PLY) * N_ACTIONS;
    for (int a = 0; a < N_ACTIONS; a++) {
        const_plane[a] = 1.0f;
    }
}

static int state_apply(CState const *state, int action, CState *out) {
    if (action < 0 || action >= N_ACTIONS) {
        PyErr_Format(PyExc_ValueError, "illegal move %d out of range", action);
        return 0;
    }
    Bits occupied = bits_or(state->current, state->opponent);
    if (bits_get(occupied, action)) {
        PyErr_Format(PyExc_ValueError, "illegal move %d on occupied square", action);
        return 0;
    }

    memset(out, 0, sizeof(CState));
    Bits mover = state->current;
    bits_set(&mover, action);
    out->current = state->opponent;
    out->opponent = mover;
    out->move_count = state->move_count + 1;

    int new_hist_len = state->hist_len + 1;
    if (new_hist_len > HISTORY_STORED) {
        new_hist_len = HISTORY_STORED;
    }
    out->hist_len = new_hist_len;
    out->hist_current[0] = state->current;
    out->hist_opponent[0] = state->opponent;
    for (int i = 1; i < new_hist_len; i++) {
        out->hist_current[i] = state->hist_current[i - 1];
        out->hist_opponent[i] = state->hist_opponent[i - 1];
    }
    return 1;
}

static int arena_reserve(NativeMCTSGameObject *game, int min_cap) {
    if (game->node_cap >= min_cap) {
        return 1;
    }
    int new_cap = game->node_cap > 0 ? game->node_cap : 1024;
    while (new_cap < min_cap) {
        new_cap *= 2;
    }
    CNode *new_nodes = (CNode *)realloc(game->nodes, sizeof(CNode) * (size_t)new_cap);
    if (new_nodes == NULL) {
        PyErr_NoMemory();
        return 0;
    }
    game->nodes = new_nodes;
    game->node_cap = new_cap;
    return 1;
}

static void init_node_fields(CNode *node, CState const *state, int parent, int parent_action) {
    memset(node, 0, sizeof(CNode));
    node->state = *state;
    node->parent = parent;
    node->parent_action = parent_action;
    for (int a = 0; a < N_ACTIONS; a++) {
        node->child[a] = -1;
    }
    if (has_five_bits(state->opponent)) {
        node->is_terminal = 1;
        node->terminal_value = -1.0f;
        return;
    }
    if (state->move_count >= N_ACTIONS) {
        node->is_terminal = 1;
        node->terminal_value = 0.0f;
        return;
    }
    Bits occupied = bits_or(state->current, state->opponent);
    int any_legal = 0;
    for (int a = 0; a < N_ACTIONS; a++) {
        node->legal[a] = (unsigned char)(!bits_get(occupied, a));
        if (node->legal[a]) {
            any_legal = 1;
        }
    }
    // Belt-and-suspenders: a position with no legal moves (whole board
    // occupied without 5-in-a-row) is a draw. Without this, the MCTS
    // selector would later try to play action 0 from a fully-occupied
    // node and crash state_apply. Hit on WL5 from archive positions where
    // the WL4 buffer's missing ply tags zero-filled, causing
    // `move_count >= N_ACTIONS` to fail to fire even on full boards.
    if (!any_legal) {
        node->is_terminal = 1;
        node->terminal_value = 0.0f;
    }
}

static int arena_add_node(
    NativeMCTSGameObject *game,
    CState const *state,
    int parent,
    int parent_action
) {
    if (!arena_reserve(game, game->node_count + 1)) {
        return -1;
    }
    int idx = game->node_count++;
    init_node_fields(&game->nodes[idx], state, parent, parent_action);
    return idx;
}

static int copy_subtree_recursive(
    CNode const *old_nodes,
    int old_idx,
    CNode *new_nodes,
    int *new_count,
    int parent,
    int parent_action,
    int max_nodes
) {
    if (*new_count >= max_nodes) {
        return -1;
    }
    int new_idx = (*new_count)++;
    new_nodes[new_idx] = old_nodes[old_idx];
    new_nodes[new_idx].parent = parent;
    new_nodes[new_idx].parent_action = parent_action;
    for (int a = 0; a < N_ACTIONS; a++) {
        int old_child = old_nodes[old_idx].child[a];
        new_nodes[new_idx].child[a] = -1;
        if (old_child >= 0) {
            int new_child = copy_subtree_recursive(
                old_nodes, old_child, new_nodes, new_count, new_idx, a, max_nodes);
            if (new_child < 0) {
                return -1;
            }
            new_nodes[new_idx].child[a] = new_child;
        }
    }
    return new_idx;
}

static int compact_to_root(NativeMCTSGameObject *game, int old_root) {
    int cap = game->node_count > 1024 ? game->node_count : 1024;
    CNode *new_nodes = (CNode *)malloc(sizeof(CNode) * (size_t)cap);
    if (new_nodes == NULL) {
        PyErr_NoMemory();
        return 0;
    }
    int new_count = 0;
    int new_root = copy_subtree_recursive(
        game->nodes, old_root, new_nodes, &new_count, -1, -1, cap);
    if (new_root < 0) {
        free(new_nodes);
        PyErr_SetString(PyExc_RuntimeError, "failed to compact native MCTS tree");
        return 0;
    }
    free(game->nodes);
    game->nodes = new_nodes;
    game->node_cap = cap;
    game->node_count = new_count;
    game->root = new_root;
    return 1;
}

static int pending_vec_init(PendingVec *vec, int cap) {
    vec->items = (Pending *)malloc(sizeof(Pending) * (size_t)cap);
    if (vec->items == NULL) {
        PyErr_NoMemory();
        return 0;
    }
    vec->count = 0;
    vec->cap = cap;
    return 1;
}

static void pending_vec_free(PendingVec *vec) {
    free(vec->items);
    vec->items = NULL;
    vec->count = 0;
    vec->cap = 0;
}

static int pending_vec_append(PendingVec *vec, Pending const *pending) {
    if (vec->count >= vec->cap) {
        int new_cap = vec->cap > 0 ? vec->cap * 2 : 64;
        Pending *new_items = (Pending *)realloc(vec->items, sizeof(Pending) * (size_t)new_cap);
        if (new_items == NULL) {
            PyErr_NoMemory();
            return 0;
        }
        vec->items = new_items;
        vec->cap = new_cap;
    }
    vec->items[vec->count++] = *pending;
    return 1;
}

static int node_total_visits(CNode const *node) {
    int total = 0;
    for (int a = 0; a < N_ACTIONS; a++) {
        total += node->N[a];
    }
    return total;
}

static int select_action(CNode const *node, double c_puct_init, double c_puct_base) {
    int total = node_total_visits(node);
    double pb_c = log((1.0 + (double)total + c_puct_base) / c_puct_base) + c_puct_init;
    double sqrt_total = sqrt((double)total + 1e-8);
    double best_score = -INFINITY;
    // Default to the first legal action so a NaN/-INF score sweep across all
    // legal moves can't leave us pointing at an illegal action (which used
    // to bite WL5 archive-start: model returned NaN value for an archived
    // mid-game position, NaN accumulated in W, all scores became NaN, the
    // old default best_action=0 then crashed state_apply with "illegal move 0
    // on occupied square" whenever square (0,0) was occupied).
    int best_action = -1;
    for (int a = 0; a < N_ACTIONS; a++) {
        if (!node->legal[a]) {
            continue;
        }
        if (best_action < 0) {
            best_action = a;
        }
        double q = node->N[a] > 0 ? (double)node->W[a] / (double)node->N[a] : 0.0;
        double u = pb_c * (double)node->P[a] * sqrt_total / (1.0 + (double)node->N[a]);
        double score = q + u;
        if (score > best_score) {
            best_score = score;
            best_action = a;
        }
    }
    if (best_action < 0) {
        best_action = 0;  // truly terminal; caller should not reach here
    }
    return best_action;
}

// KataGo forced-playout count for one root child (Wu 2019, "Accelerating
// Self-Play Learning in Go"). n_forced(a) = ceil(sqrt(k * P(a) * N_root)).
// N_root is the total visit count over the root's children.
static inline int n_forced_visits(double k, double prior, int root_total) {
    if (k <= 0.0 || prior <= 0.0 || root_total <= 0) {
        return 0;
    }
    double q = sqrt(k * prior * (double)root_total);
    double c = ceil(q);
    if (c < 0.0) {
        c = 0.0;
    }
    return (int)c;
}

// Root-only forced-playout override. Returns a legal action to force, or -1
// if no child is under its forced quota (caller falls back to PUCT). We only
// force children that have ALREADY been visited at least once (N[a] > 0) and
// are below their forced quota — matching KataGo, which tops up explored
// children rather than seeding every legal move. Among eligible children we
// pick the one with the largest forced-vs-actual deficit, breaking ties by
// prior, so the selection is deterministic and reproducible.
static int select_forced_root_action(CNode const *node, double k) {
    int total = node_total_visits(node);
    if (total <= 0) {
        return -1;
    }
    int best_action = -1;
    int best_deficit = 0;
    double best_prior = -1.0;
    for (int a = 0; a < N_ACTIONS; a++) {
        if (!node->legal[a]) {
            continue;
        }
        if (node->N[a] <= 0) {
            continue;  // only force already-explored children
        }
        int nf = n_forced_visits(k, (double)node->P[a], total);
        int deficit = nf - node->N[a];
        if (deficit <= 0) {
            continue;
        }
        if (deficit > best_deficit ||
            (deficit == best_deficit && (double)node->P[a] > best_prior)) {
            best_deficit = deficit;
            best_prior = (double)node->P[a];
            best_action = a;
        }
    }
    return best_action;
}

static int select_one_vloss(NativeMCTSGameObject *game, Pending *out) {
    int node_idx = game->root;
    out->path_len = 0;
    for (;;) {
        CNode *node = &game->nodes[node_idx];
        if (node->is_terminal || !node->expanded) {
            out->leaf_index = node_idx;
            return 1;
        }
        int action = -1;
        // Forced playouts apply at the ROOT only (KataGo). Default k==0.0
        // disables this entirely so behavior is byte-identical to legacy.
        if (game->forced_playout_k > 0.0 && node_idx == game->root) {
            action = select_forced_root_action(node, game->forced_playout_k);
        }
        if (action < 0) {
            action = select_action(node, game->c_puct, game->c_puct_base);
        }
        int child_idx = node->child[action];
        if (child_idx < 0) {
            CState child_state;
            if (!state_apply(&node->state, action, &child_state)) {
                return 0;
            }
            child_idx = arena_add_node(game, &child_state, node_idx, action);
            if (child_idx < 0) {
                return 0;
            }
            node = &game->nodes[node_idx];
            node->child[action] = child_idx;
        }
        if (out->path_len >= MAX_PATH) {
            PyErr_SetString(PyExc_RuntimeError, "native MCTS path exceeded board size");
            return 0;
        }
        out->path_nodes[out->path_len] = node_idx;
        out->path_actions[out->path_len] = action;
        out->path_len++;
        node->N[action] += 1;
        node_idx = child_idx;
    }
}

// Gumbel descent: the FIRST edge from the root is FORCED to
// game->gumbel_forced_action (the Sequential-Halving-scheduled candidate);
// every node below the root uses ordinary PUCT (identical to
// select_one_vloss). Mirrors gumbel._simulate_from_child. Virtual loss
// (N[action]+=1) is applied on the forced root edge exactly as PUCT would,
// so the wave-batching / vloss bookkeeping is unchanged.
static int select_one_vloss_gumbel(NativeMCTSGameObject *game, Pending *out) {
    out->path_len = 0;
    CNode *root = &game->nodes[game->root];
    if (root->is_terminal || !root->expanded) {
        out->leaf_index = game->root;
        return 1;
    }
    int forced = game->gumbel_forced_action;
    if (forced < 0 || forced >= N_ACTIONS || !root->legal[forced]) {
        // No valid forced action this slot => contribute nothing (leaf=root,
        // terminal handling in the caller treats an unexpanded/forced-less
        // slot specially; we guard by never enqueueing such a slot). Fall
        // back to PUCT-from-root to stay safe.
        return select_one_vloss(game, out);
    }
    int child_idx = root->child[forced];
    if (child_idx < 0) {
        CState child_state;
        if (!state_apply(&root->state, forced, &child_state)) {
            return 0;
        }
        child_idx = arena_add_node(game, &child_state, game->root, forced);
        if (child_idx < 0) {
            return 0;
        }
        root = &game->nodes[game->root];
        root->child[forced] = child_idx;
    }
    out->path_nodes[out->path_len] = game->root;
    out->path_actions[out->path_len] = forced;
    out->path_len++;
    root->N[forced] += 1;

    // Descend from the child with ordinary PUCT.
    int node_idx = child_idx;
    for (;;) {
        CNode *node = &game->nodes[node_idx];
        if (node->is_terminal || !node->expanded) {
            out->leaf_index = node_idx;
            return 1;
        }
        int action = select_action(node, game->c_puct, game->c_puct_base);
        int next_child = node->child[action];
        if (next_child < 0) {
            CState child_state;
            if (!state_apply(&node->state, action, &child_state)) {
                return 0;
            }
            next_child = arena_add_node(game, &child_state, node_idx, action);
            if (next_child < 0) {
                return 0;
            }
            node = &game->nodes[node_idx];
            node->child[action] = next_child;
        }
        if (out->path_len >= MAX_PATH) {
            PyErr_SetString(PyExc_RuntimeError, "native MCTS path exceeded board size");
            return 0;
        }
        out->path_nodes[out->path_len] = node_idx;
        out->path_actions[out->path_len] = action;
        out->path_len++;
        node->N[action] += 1;
        node_idx = next_child;
    }
}

static void backprop_value_only(NativeMCTSGameObject *game, Pending const *pending, float leaf_value) {
    float v = leaf_value;
    for (int i = pending->path_len - 1; i >= 0; i--) {
        v = -v;
        CNode *parent = &game->nodes[pending->path_nodes[i]];
        int action = pending->path_actions[i];
        parent->W[action] += v;
    }
}

static void set_priors(CNode *node, float const *raw_priors) {
    double max_val = -INFINITY;
    int legal_count = 0;
    for (int a = 0; a < N_ACTIONS; a++) {
        if (node->legal[a]) {
            legal_count++;
            if ((double)raw_priors[a] > max_val) {
                max_val = (double)raw_priors[a];
            }
        }
    }
    if (legal_count == 0) {
        return;
    }

    // Gumbel needs the *centered* raw logits (legal: x - legal_max; illegal:
    // -INF). Mirrors gumbel.py::_root_logits. This is computed for every node
    // but only consumed at the root in the Gumbel path; it costs one extra
    // store per legal action and is otherwise inert (PUCT never reads logits).
    for (int a = 0; a < N_ACTIONS; a++) {
        node->logits[a] = node->legal[a]
            ? (float)((double)raw_priors[a] - max_val)
            : -INFINITY;
    }

    double sum = 0.0;
    for (int a = 0; a < N_ACTIONS; a++) {
        if (node->legal[a]) {
            double e = exp((double)raw_priors[a] - max_val);
            node->P[a] = (float)e;
            sum += e;
        } else {
            node->P[a] = 0.0f;
        }
    }
    if (sum > 0.0 && isfinite(sum)) {
        for (int a = 0; a < N_ACTIONS; a++) {
            node->P[a] = node->legal[a] ? (float)((double)node->P[a] / sum) : 0.0f;
        }
    } else {
        float p = 1.0f / (float)legal_count;
        for (int a = 0; a < N_ACTIONS; a++) {
            node->P[a] = node->legal[a] ? p : 0.0f;
        }
    }
}

static void add_dirichlet_noise(NativeMCTSGameObject *game, CNode *node) {
    if (game->dirichlet_eps <= 0.0 || game->dirichlet_alpha <= 0.0) {
        return;
    }
    double noise[N_ACTIONS];
    double sum = 0.0;
    int legal_count = 0;
    for (int a = 0; a < N_ACTIONS; a++) {
        noise[a] = 0.0;
        if (node->legal[a]) {
            double x = rng_gamma(game, game->dirichlet_alpha);
            noise[a] = x;
            sum += x;
            legal_count++;
        }
    }
    if (legal_count == 0 || sum <= 0.0 || !isfinite(sum)) {
        return;
    }
    for (int a = 0; a < N_ACTIONS; a++) {
        if (node->legal[a]) {
            double n = noise[a] / sum;
            node->P[a] = (float)((1.0 - game->dirichlet_eps) * (double)node->P[a]
                                + game->dirichlet_eps * n);
        }
    }
}

// ===========================================================================
// Gumbel AlphaZero root selection + Sequential Halving (Danihelka et al. 2022)
// Mirrors gomoku/gumbel.py. Root-only; internal nodes keep PUCT. See the
// C-port spec at the bottom of gomoku/gumbel.py.
// ===========================================================================

// sigma(q) = (c_visit + max_visit_at_root) * c_scale * q. Matches gumbel._sigma.
static inline double sigma_q(double q, int max_visit, double c_visit, double c_scale) {
    return (c_visit + (double)max_visit) * c_scale * q;
}

// Draw gumbel_noise[a] for each legal a, score = g + logits, and keep the top
// min(m, n_legal) legal actions (descending by score) into gumbel_candidates.
// Mirrors gumbel._gumbel_topk (selection-sort to make the descending order
// deterministic and tie-stable like numpy argsort[::-1] on distinct floats).
static void gumbel_sample_topk(NativeMCTSGameObject *game, CNode *root, int m) {
    double scores[N_ACTIONS];
    int legal_idx[N_ACTIONS];
    int n_legal = 0;
    for (int a = 0; a < N_ACTIONS; a++) {
        game->gumbel_noise[a] = 0.0;
        if (root->legal[a]) {
            double g = rng_gumbel(game);
            game->gumbel_noise[a] = g;
            scores[a] = g + (double)root->logits[a];
            legal_idx[n_legal++] = a;
        } else {
            scores[a] = -INFINITY;
        }
    }
    int k = m < n_legal ? m : n_legal;
    if (k < 0) {
        k = 0;
    }
    // Partial selection sort: pick the top-k by score from legal_idx.
    for (int i = 0; i < k; i++) {
        int best = i;
        for (int j = i + 1; j < n_legal; j++) {
            if (scores[legal_idx[j]] > scores[legal_idx[best]]) {
                best = j;
            }
        }
        int tmp = legal_idx[i];
        legal_idx[i] = legal_idx[best];
        legal_idx[best] = tmp;
        game->gumbel_candidates[i] = legal_idx[i];
    }
    game->gumbel_n_candidates = k;
    game->gumbel_sampled = 1;
}

// Score of a single root candidate: g(a) + logits(a) + sigma(q_hat(a)).
// q_hat(a) = W[a]/N[a] for visited, else 0.0 (matches gumbel.py's SH/argmax
// scoring, which uses q=0 for unvisited survivors).
static double gumbel_candidate_score(
    NativeMCTSGameObject *game, CNode *root, int a, int max_visit
) {
    int n = root->N[a];
    double q = n > 0 ? (double)root->W[a] / (double)n : 0.0;
    return game->gumbel_noise[a] + (double)root->logits[a]
         + sigma_q(q, max_visit, game->gumbel_c_visit, game->gumbel_c_scale);
}

static int gumbel_root_max_visit(CNode *root) {
    int max_visit = 0;
    for (int a = 0; a < N_ACTIONS; a++) {
        if (root->legal[a] && root->N[a] > max_visit) {
            max_visit = root->N[a];
        }
    }
    return max_visit;
}

// Sequential Halving per-phase visits-per-candidate. Mirrors
// gumbel._sequential_halving_schedule exactly.
static int gumbel_sh_schedule(int m, int budget, int *out, int max_phases) {
    if (m <= 1 || budget <= 0) {
        return 0;
    }
    int n_phases = (int)floor(log2((double)m));
    if (n_phases < 1) {
        n_phases = 1;
    }
    if (n_phases > max_phases) {
        n_phases = max_phases;
    }
    int sizes[N_ACTIONS];
    int cur = m;
    for (int i = 0; i < n_phases; i++) {
        sizes[i] = cur;
        cur = cur / 2;
        if (cur < 2) {
            cur = 2;
        }
    }
    int per_phase_total = budget / n_phases;
    if (per_phase_total < 1) {
        per_phase_total = 1;
    }
    int used = 0;
    for (int i = 0; i < n_phases; i++) {
        int n_per;
        if (i == n_phases - 1) {
            int remaining = budget - used;
            n_per = remaining / sizes[i];
            if (n_per < 1) {
                n_per = 1;
            }
        } else {
            n_per = per_phase_total / sizes[i];
            if (n_per < 1) {
                n_per = 1;
            }
        }
        out[i] = n_per;
        used += n_per * sizes[i];
    }
    return n_phases;
}

// Final selection: argmax of g+logits+sigma(q_hat) over current survivors.
// Mirrors gumbel.py step 4 (first survivor as the initial best, strict > to
// break ties toward the higher-scoring sampled order).
static int gumbel_select_score_argmax(NativeMCTSGameObject *game, CNode *root) {
    if (game->gumbel_n_candidates <= 0) {
        return -1;
    }
    int max_visit = gumbel_root_max_visit(root);
    int best_a = game->gumbel_candidates[0];
    double best_score = -INFINITY;
    for (int i = 0; i < game->gumbel_n_candidates; i++) {
        int a = game->gumbel_candidates[i];
        double s = gumbel_candidate_score(game, root, a, max_visit);
        if (s > best_score) {
            best_score = s;
            best_a = a;
        }
    }
    return best_a;
}

// Keep the top half (by score) of the current survivors in place. Mirrors
// gumbel.py: keep = max(1, len//2); top = argsort(scores)[::-1][:keep].
static void gumbel_halve_survivors(NativeMCTSGameObject *game, CNode *root) {
    int n = game->gumbel_n_candidates;
    if (n <= 1) {
        return;
    }
    int max_visit = gumbel_root_max_visit(root);
    double scores[N_ACTIONS];
    for (int i = 0; i < n; i++) {
        scores[i] = gumbel_candidate_score(game, root, game->gumbel_candidates[i], max_visit);
    }
    int keep = n / 2;
    if (keep < 1) {
        keep = 1;
    }
    // Partial selection sort of candidates by score desc, keep top `keep`.
    for (int i = 0; i < keep; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) {
            if (scores[j] > scores[best]) {
                best = j;
            }
        }
        double st = scores[i];
        scores[i] = scores[best];
        scores[best] = st;
        int ct = game->gumbel_candidates[i];
        game->gumbel_candidates[i] = game->gumbel_candidates[best];
        game->gumbel_candidates[best] = ct;
    }
    game->gumbel_n_candidates = keep;
}

static int call_evaluator(
    PyObject *evaluator,
    CNode **leaf_nodes,
    int n,
    PyArrayObject **priors_out,
    PyArrayObject **values_out
) {
    npy_intp dims[4] = {n, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE};
    PyObject *batch_obj = PyArray_SimpleNew(4, dims, NPY_FLOAT32);
    if (batch_obj == NULL) {
        return 0;
    }
    float *batch = (float *)PyArray_DATA((PyArrayObject *)batch_obj);
    for (int i = 0; i < n; i++) {
        state_to_planes(&leaf_nodes[i]->state, batch + (size_t)i * N_INPUT_PLANES * N_ACTIONS);
    }

    PyObject *result = PyObject_CallFunctionObjArgs(evaluator, batch_obj, NULL);
    Py_DECREF(batch_obj);
    if (result == NULL) {
        return 0;
    }
    if (!PyTuple_Check(result) || PyTuple_GET_SIZE(result) != 2) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_TypeError, "native MCTS evaluator must return (priors, values)");
        return 0;
    }

    PyArrayObject *priors = (PyArrayObject *)PyArray_FROM_OTF(
        PyTuple_GET_ITEM(result, 0), NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values = (PyArrayObject *)PyArray_FROM_OTF(
        PyTuple_GET_ITEM(result, 1), NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    Py_DECREF(result);
    if (priors == NULL || values == NULL) {
        Py_XDECREF(priors);
        Py_XDECREF(values);
        return 0;
    }
    if (PyArray_NDIM(priors) != 2 || PyArray_DIMS(priors)[0] != n ||
        PyArray_DIMS(priors)[1] != N_ACTIONS) {
        Py_DECREF(priors);
        Py_DECREF(values);
        PyErr_SetString(PyExc_ValueError, "native MCTS priors must have shape (B, 81)");
        return 0;
    }
    if (PyArray_NDIM(values) != 1 || PyArray_DIMS(values)[0] != n) {
        Py_DECREF(priors);
        Py_DECREF(values);
        PyErr_SetString(PyExc_ValueError, "native MCTS values must have shape (B,)");
        return 0;
    }

    *priors_out = priors;
    *values_out = values;
    return 1;
}

static int evaluate_and_expand(
    PyObject *evaluator,
    CNode **leaf_nodes,
    int n,
    NativeMCTSGameObject **noise_games,
    int add_noise
) {
    if (n <= 0) {
        return 1;
    }
    PyArrayObject *priors = NULL;
    PyArrayObject *values = NULL;
    if (!call_evaluator(evaluator, leaf_nodes, n, &priors, &values)) {
        return 0;
    }
    float const *prior_data = (float const *)PyArray_DATA(priors);
    for (int i = 0; i < n; i++) {
        CNode *node = leaf_nodes[i];
        if (!node->expanded) {
            set_priors(node, prior_data + (size_t)i * N_ACTIONS);
            node->expanded = 1;
            if (add_noise && noise_games != NULL && noise_games[i] != NULL) {
                add_dirichlet_noise(noise_games[i], node);
            }
        }
    }
    Py_DECREF(priors);
    Py_DECREF(values);
    return 1;
}

// Gumbel root expansion: ALWAYS evaluates the root batch so we can capture the
// network's root value (for v_mix completion) and the centered logits, even
// when the root was already expanded via subtree reuse. Mirrors gumbel.py's
// gumbel_search_root step 1, which re-evaluates the root every call but only
// _set_priors when unexpanded. `games[i]` are the games whose roots are in
// `root_leaves[i]` (parallel arrays).
static int evaluate_and_expand_root_gumbel(
    PyObject *evaluator,
    CNode **root_leaves,
    NativeMCTSGameObject **owner_games,
    int n
) {
    if (n <= 0) {
        return 1;
    }
    PyArrayObject *priors = NULL;
    PyArrayObject *values = NULL;
    if (!call_evaluator(evaluator, root_leaves, n, &priors, &values)) {
        return 0;
    }
    float const *prior_data = (float const *)PyArray_DATA(priors);
    float const *value_data = (float const *)PyArray_DATA(values);
    for (int i = 0; i < n; i++) {
        CNode *node = root_leaves[i];
        owner_games[i]->root_value = value_data[i];
        if (!node->expanded) {
            set_priors(node, prior_data + (size_t)i * N_ACTIONS);
            node->expanded = 1;
            // Gumbel uses Gumbel noise (drawn in gumbel_sample_topk), NOT
            // Dirichlet, so no add_dirichlet_noise here.
        } else {
            // Subtree-reuse root: P/logits already set by the leaf expansion
            // that created it. Re-derive the centered logits from the freshly
            // evaluated raw priors so they match what gumbel.py would compute
            // for THIS ply (the prior P is left untouched for v_mix weights /
            // PUCT below root, matching the Python path which also leaves an
            // already-expanded root's P intact).
            float const *raw = prior_data + (size_t)i * N_ACTIONS;
            double max_val = -INFINITY;
            for (int a = 0; a < N_ACTIONS; a++) {
                if (node->legal[a] && (double)raw[a] > max_val) {
                    max_val = (double)raw[a];
                }
            }
            for (int a = 0; a < N_ACTIONS; a++) {
                node->logits[a] = node->legal[a]
                    ? (float)((double)raw[a] - max_val)
                    : -INFINITY;
            }
        }
    }
    Py_DECREF(priors);
    Py_DECREF(values);
    return 1;
}

static void NativeMCTSGame_dealloc(NativeMCTSGameObject *self) {
    free(self->nodes);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int NativeMCTSGame_init(
    NativeMCTSGameObject *self,
    PyObject *args,
    PyObject *kwargs
) {
    PyObject *state_obj = Py_None;
    unsigned long long seed = 1;
    self->c_puct = 1.25;
    self->c_puct_base = 19652.0;
    self->dirichlet_alpha = 0.3;
    self->dirichlet_eps = 0.25;
    self->forced_playout_k = 0.0;  // OFF by default (byte-identical legacy)
    // Gumbel defaults: OFF (PUCT). When gumbel_root != 0, the root edge is
    // forced via Sequential Halving; m/c_visit/c_scale follow the paper.
    int gumbel_root = 0;
    int gumbel_m = 16;
    double gumbel_c_visit = 50.0;
    double gumbel_c_scale = 1.0;
    static char *kwlist[] = {
        "state", "c_puct", "c_puct_base", "dirichlet_alpha", "dirichlet_eps", "seed",
        "forced_playout_k", "gumbel_root", "gumbel_m", "gumbel_c_visit", "gumbel_c_scale", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "|OddddKdiidd", kwlist,
            &state_obj,
            &self->c_puct,
            &self->c_puct_base,
            &self->dirichlet_alpha,
            &self->dirichlet_eps,
            &seed,
            &self->forced_playout_k,
            &gumbel_root,
            &gumbel_m,
            &gumbel_c_visit,
            &gumbel_c_scale)) {
        return -1;
    }

    self->nodes = NULL;
    self->node_count = 0;
    self->node_cap = 0;
    self->root = -1;
    self->rng = seed == 0 ? 1 : (uint64_t)seed;
    self->gumbel_root = gumbel_root;
    self->gumbel_m = gumbel_m;
    self->gumbel_c_visit = gumbel_c_visit;
    self->gumbel_c_scale = gumbel_c_scale;
    self->gumbel_n_candidates = 0;
    self->root_value = 0.0f;
    self->gumbel_forced_action = -1;
    self->gumbel_selected_action = -1;
    self->gumbel_sampled = 0;

    CState state;
    if (state_obj == Py_None) {
        state_initial(&state);
    } else if (!state_from_py(state_obj, &state)) {
        return -1;
    }
    int root = arena_add_node(self, &state, -1, -1);
    if (root < 0) {
        return -1;
    }
    self->root = root;
    return 0;
}

// KataGo policy-target pruning (Wu 2019). Fill `pruned` with the root child
// visit counts AFTER subtracting forced playouts. The most-visited child keeps
// all its visits; for each other child we subtract as many visits as possible
// (up to its n_forced quota) WITHOUT letting its PUCT exploration-selection
// value exceed the best child's — i.e. we remove only the artificially-forced
// exploration, never visits PUCT would have spent on its own. With k <= 0 this
// is a straight copy of root->N (byte-identical legacy target).
static void compute_pruned_root_visits(
    NativeMCTSGameObject const *game, CNode const *root, double k, int32_t *pruned
) {
    for (int a = 0; a < N_ACTIONS; a++) {
        pruned[a] = root->N[a];
    }
    if (k <= 0.0) {
        return;
    }
    int total = node_total_visits(root);
    if (total <= 0) {
        return;
    }
    // Most-visited (best) child keeps all its visits.
    int best_action = -1;
    int best_count = -1;
    for (int a = 0; a < N_ACTIONS; a++) {
        if (!root->legal[a]) {
            continue;
        }
        if (root->N[a] > best_count) {
            best_count = root->N[a];
            best_action = a;
        }
    }
    if (best_action < 0) {
        return;
    }
    // AGZ log-schedule PUCT coefficient (same as select_action). Visit-count
    // pruning is a target-extraction step; we use the FINAL parent total here.
    double pb_c = log((1.0 + (double)total + game->c_puct_base) / game->c_puct_base)
                  + game->c_puct;
    double sqrt_total = sqrt((double)total + 1e-8);

    // Best child's PUCT exploration-selection value (Q + U) at its real visits.
    double best_q = root->N[best_action] > 0
        ? (double)root->W[best_action] / (double)root->N[best_action]
        : 0.0;
    double best_u = pb_c * (double)root->P[best_action] * sqrt_total
                    / (1.0 + (double)root->N[best_action]);
    double best_value = best_q + best_u;

    for (int a = 0; a < N_ACTIONS; a++) {
        if (a == best_action || !root->legal[a] || root->N[a] <= 0) {
            continue;
        }
        int nf = n_forced_visits(k, (double)root->P[a], total);
        if (nf <= 0) {
            continue;
        }
        // Q is fixed by W/N at the real visit count (KataGo holds the child's
        // value constant while shrinking its U via the (1 + n) denominator).
        double q = (double)root->W[a] / (double)root->N[a];
        int n = root->N[a];
        int subtracted = 0;
        // Remove forced visits one at a time, up to nf, but stop the moment
        // the child's selection value at the reduced count would meet/exceed
        // the best child's (i.e. PUCT would have picked it on its own).
        while (subtracted < nf && n > 1) {
            double u_at = pb_c * (double)root->P[a] * sqrt_total / (1.0 + (double)(n - 1));
            if (q + u_at > best_value) {
                break;  // removing this visit would make the child PUCT-preferred
            }
            n -= 1;
            subtracted += 1;
        }
        pruned[a] = n;
    }
}

static PyObject *NativeMCTSGame_policy(NativeMCTSGameObject *self, PyObject *args, PyObject *kwargs) {
    double temperature = 1.0;
    double forced_playout_k = -1.0;  // <0 sentinel: fall back to game's stored k
    static char *kwlist[] = {"temperature", "forced_playout_k", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|dd", kwlist, &temperature, &forced_playout_k)) {
        return NULL;
    }
    npy_intp dims[1] = {N_ACTIONS};
    PyObject *out_obj = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (out_obj == NULL) {
        return NULL;
    }
    float *out = (float *)PyArray_DATA((PyArrayObject *)out_obj);
    CNode *root = &self->nodes[self->root];
    double k = forced_playout_k >= 0.0 ? forced_playout_k : self->forced_playout_k;
    int32_t pruned[N_ACTIONS];
    compute_pruned_root_visits(self, root, k, pruned);

    if (temperature <= 0.0) {
        int max_count = pruned[0];
        for (int a = 1; a < N_ACTIONS; a++) {
            if (pruned[a] > max_count) {
                max_count = pruned[a];
            }
        }
        int winners = 0;
        for (int a = 0; a < N_ACTIONS; a++) {
            if (pruned[a] == max_count) {
                winners++;
            }
        }
        float p = winners > 0 ? 1.0f / (float)winners : 0.0f;
        for (int a = 0; a < N_ACTIONS; a++) {
            out[a] = pruned[a] == max_count ? p : 0.0f;
        }
        return out_obj;
    }

    double sum = 0.0;
    if (temperature == 1.0) {
        for (int a = 0; a < N_ACTIONS; a++) {
            sum += (double)pruned[a];
        }
        if (sum > 0.0) {
            for (int a = 0; a < N_ACTIONS; a++) {
                out[a] = (float)((double)pruned[a] / sum);
            }
        } else {
            int legal_count = 0;
            for (int a = 0; a < N_ACTIONS; a++) {
                legal_count += root->legal[a] ? 1 : 0;
            }
            float p = legal_count > 0 ? 1.0f / (float)legal_count : 0.0f;
            for (int a = 0; a < N_ACTIONS; a++) {
                out[a] = root->legal[a] ? p : 0.0f;
            }
        }
        return out_obj;
    }

    double inv_temp = 1.0 / temperature;
    // Keep per-action sharpened scores in DOUBLE to avoid float32 overflow.
    // pow(N, 1/tau) can exceed FLT_MAX (~3.4e38) when N is large and tau is
    // small — e.g. at WL3's tau=0.1 with subtree-reused root visits of
    // ~7200+ (which happens routinely past 18 plies in concentrated games),
    // pow(7200, 10) ≈ 3.7e38 saturates float32 to +Inf. The subsequent
    // out[a] = (float)out[a]/sum then carries that Inf forward, sum is
    // +Inf, and the Python sampler hits Inf/Inf → NaN. Doing all
    // arithmetic in double and only casting the *normalized* (in [0,1])
    // probabilities to float32 keeps the output strictly finite.
    double scores[N_ACTIONS];
    for (int a = 0; a < N_ACTIONS; a++) {
        double x = pow((double)pruned[a], inv_temp);
        scores[a] = x;
        sum += x;
    }
    if (sum > 0.0 && isfinite(sum)) {
        for (int a = 0; a < N_ACTIONS; a++) {
            out[a] = (float)(scores[a] / sum);
        }
    } else if (isfinite(sum)) {
        int legal_count = 0;
        for (int a = 0; a < N_ACTIONS; a++) {
            legal_count += root->legal[a] ? 1 : 0;
        }
        float p = legal_count > 0 ? 1.0f / (float)legal_count : 0.0f;
        for (int a = 0; a < N_ACTIONS; a++) {
            out[a] = root->legal[a] ? p : 0.0f;
        }
    } else {
        // sum overflowed double (only possible at absurd tau or N). Fall back
        // to argmax-tie distribution so the policy is still a valid pmf.
        int32_t max_count = pruned[0];
        for (int a = 1; a < N_ACTIONS; a++) {
            if (pruned[a] > max_count) {
                max_count = pruned[a];
            }
        }
        int winners = 0;
        for (int a = 0; a < N_ACTIONS; a++) {
            if (pruned[a] == max_count) {
                winners++;
            }
        }
        float p = winners > 0 ? 1.0f / (float)winners : 0.0f;
        for (int a = 0; a < N_ACTIONS; a++) {
            out[a] = pruned[a] == max_count ? p : 0.0f;
        }
    }
    return out_obj;
}

static PyObject *NativeMCTSGame_visit_counts(NativeMCTSGameObject *self, PyObject *Py_UNUSED(ignored)) {
    npy_intp dims[1] = {N_ACTIONS};
    PyObject *out_obj = PyArray_SimpleNew(1, dims, NPY_INT32);
    if (out_obj == NULL) {
        return NULL;
    }
    int32_t *out = (int32_t *)PyArray_DATA((PyArrayObject *)out_obj);
    memcpy(out, self->nodes[self->root].N, N_ACTIONS * sizeof(int32_t));
    return out_obj;
}

// Gumbel completed-policy TARGET. Mirrors gumbel.py::_root_q_completed +
// completed_policy_target EXACTLY:
//   q[a] = W[a]/N[a] for visited (already root-perspective via backprop);
//   v_mix = (root_value + (sum_N/sum_P_visited)*sum(P[a]*q[a])) / (1+sum_N);
//   q[unvisited] = v_mix;
//   target = softmax over legal of logits[a] + sigma_q(q[a], maxN).
static PyObject *NativeMCTSGame_gumbel_policy(NativeMCTSGameObject *self, PyObject *args, PyObject *kwargs) {
    double c_visit = self->gumbel_c_visit;
    double c_scale = self->gumbel_c_scale;
    static char *kwlist[] = {"c_visit", "c_scale", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|dd", kwlist, &c_visit, &c_scale)) {
        return NULL;
    }
    npy_intp dims[1] = {N_ACTIONS};
    PyObject *out_obj = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (out_obj == NULL) {
        return NULL;
    }
    float *out = (float *)PyArray_DATA((PyArrayObject *)out_obj);
    for (int a = 0; a < N_ACTIONS; a++) {
        out[a] = 0.0f;
    }
    CNode *root = &self->nodes[self->root];

    int n_legal = 0;
    int max_visit = 0;
    long sum_n = 0;
    for (int a = 0; a < N_ACTIONS; a++) {
        if (root->legal[a]) {
            n_legal++;
            if (root->N[a] > max_visit) {
                max_visit = root->N[a];
            }
            sum_n += root->N[a];
        }
    }
    if (n_legal == 0) {
        return out_obj;  // no legal actions; all-zero target
    }

    // Completed Q (v_mix for unvisited). q[visited] = W/N.
    double q[N_ACTIONS];
    double weighted_q = 0.0;     // sum over visited of P[a]*q[a]
    double sum_pi_visited = 0.0; // sum over visited of P[a]
    for (int a = 0; a < N_ACTIONS; a++) {
        q[a] = 0.0;
        if (root->legal[a] && root->N[a] > 0) {
            q[a] = (double)root->W[a] / (double)root->N[a];
            weighted_q += (double)root->P[a] * q[a];
            sum_pi_visited += (double)root->P[a];
        }
    }
    double v_mix;
    if (sum_n > 0 && sum_pi_visited > 0.0) {
        v_mix = ((double)self->root_value + ((double)sum_n / sum_pi_visited) * weighted_q)
                / (1.0 + (double)sum_n);
    } else {
        v_mix = (double)self->root_value;
    }
    for (int a = 0; a < N_ACTIONS; a++) {
        if (root->legal[a] && root->N[a] == 0) {
            q[a] = v_mix;
        }
    }

    // raw[a] = logits[a] + sigma(q[a]); softmax over legal (illegal -> -inf).
    double raw[N_ACTIONS];
    double m = -INFINITY;
    int any_finite = 0;
    for (int a = 0; a < N_ACTIONS; a++) {
        if (root->legal[a]) {
            double sig = sigma_q(q[a], max_visit, c_visit, c_scale);
            raw[a] = (double)root->logits[a] + sig;
            if (isfinite(raw[a])) {
                any_finite = 1;
                if (raw[a] > m) {
                    m = raw[a];
                }
            }
        } else {
            raw[a] = -INFINITY;
        }
    }
    if (!any_finite) {
        float p = 1.0f / (float)n_legal;
        for (int a = 0; a < N_ACTIONS; a++) {
            out[a] = root->legal[a] ? p : 0.0f;
        }
        return out_obj;
    }
    double s = 0.0;
    double exp_a[N_ACTIONS];
    for (int a = 0; a < N_ACTIONS; a++) {
        if (root->legal[a]) {
            exp_a[a] = exp(raw[a] - m);
            s += exp_a[a];
        } else {
            exp_a[a] = 0.0;
        }
    }
    if (s > 0.0) {
        for (int a = 0; a < N_ACTIONS; a++) {
            out[a] = (float)(exp_a[a] / s);
        }
    } else {
        float p = 1.0f / (float)n_legal;
        for (int a = 0; a < N_ACTIONS; a++) {
            out[a] = root->legal[a] ? p : 0.0f;
        }
    }
    return out_obj;
}

static PyObject *NativeMCTSGame_gumbel_selected_action(NativeMCTSGameObject *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromLong(self->gumbel_selected_action);
}

// Test-only: export the root's internal Gumbel state (W/P/logits/root_value/
// noise/candidates) so the parity test can recompute the Python reference's
// completed_policy_target from the EXACT same tree state and assert agreement.
// Not used in production. Returns a dict.
static PyObject *NativeMCTSGame_gumbel_debug_state(NativeMCTSGameObject *self, PyObject *Py_UNUSED(ignored)) {
    CNode *root = &self->nodes[self->root];
    npy_intp dims[1] = {N_ACTIONS};
    PyObject *W = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *P = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *logits = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *noise = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (W == NULL || P == NULL || logits == NULL || noise == NULL) {
        Py_XDECREF(W); Py_XDECREF(P); Py_XDECREF(logits); Py_XDECREF(noise);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject *)W), root->W, N_ACTIONS * sizeof(float));
    memcpy(PyArray_DATA((PyArrayObject *)P), root->P, N_ACTIONS * sizeof(float));
    memcpy(PyArray_DATA((PyArrayObject *)logits), root->logits, N_ACTIONS * sizeof(float));
    memcpy(PyArray_DATA((PyArrayObject *)noise), self->gumbel_noise, N_ACTIONS * sizeof(double));
    PyObject *d = PyDict_New();
    if (d == NULL) {
        Py_DECREF(W); Py_DECREF(P); Py_DECREF(logits); Py_DECREF(noise);
        return NULL;
    }
    PyDict_SetItemString(d, "W", W);
    PyDict_SetItemString(d, "P", P);
    PyDict_SetItemString(d, "logits", logits);
    PyDict_SetItemString(d, "noise", noise);
    PyObject *rv = PyFloat_FromDouble((double)self->root_value);
    PyDict_SetItemString(d, "root_value", rv);
    Py_DECREF(rv);
    Py_DECREF(W); Py_DECREF(P); Py_DECREF(logits); Py_DECREF(noise);
    return d;
}

static PyObject *NativeMCTSGame_root_planes(NativeMCTSGameObject *self, PyObject *Py_UNUSED(ignored)) {
    npy_intp dims[3] = {N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE};
    PyObject *out_obj = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (out_obj == NULL) {
        return NULL;
    }
    state_to_planes(&self->nodes[self->root].state, (float *)PyArray_DATA((PyArrayObject *)out_obj));
    return out_obj;
}

static PyObject *NativeMCTSGame_advance_root(NativeMCTSGameObject *self, PyObject *args) {
    int action;
    if (!PyArg_ParseTuple(args, "i", &action)) {
        return NULL;
    }
    if (action < 0 || action >= N_ACTIONS) {
        PyErr_Format(PyExc_ValueError, "illegal move %d out of range", action);
        return NULL;
    }
    CNode *root = &self->nodes[self->root];
    int child_idx = root->child[action];
    if (child_idx >= 0) {
        if (!compact_to_root(self, child_idx)) {
            return NULL;
        }
    } else {
        CState next_state;
        if (!state_apply(&root->state, action, &next_state)) {
            return NULL;
        }
        self->node_count = 0;
        int next_root = arena_add_node(self, &next_state, -1, -1);
        if (next_root < 0) {
            return NULL;
        }
        self->root = next_root;
    }
    // Per-root Gumbel state is stale once the root moves; force a fresh
    // topk sample on the next gumbel search.
    self->gumbel_sampled = 0;
    self->gumbel_n_candidates = 0;
    self->gumbel_forced_action = -1;
    self->gumbel_selected_action = -1;
    Py_RETURN_NONE;
}

static PyObject *NativeMCTSGame_is_terminal(NativeMCTSGameObject *self, PyObject *Py_UNUSED(ignored)) {
    CNode *root = &self->nodes[self->root];
    return Py_BuildValue("Nf", PyBool_FromLong(root->is_terminal), root->terminal_value);
}

static PyObject *NativeMCTSGame_move_count(NativeMCTSGameObject *self, void *closure) {
    (void)closure;
    return PyLong_FromLong(self->nodes[self->root].state.move_count);
}

static PyGetSetDef NativeMCTSGame_getset[] = {
    {"move_count", (getter)NativeMCTSGame_move_count, NULL, "root move count", NULL},
    {NULL}
};

static PyMethodDef NativeMCTSGame_methods[] = {
    {"policy", (PyCFunction)NativeMCTSGame_policy, METH_VARARGS | METH_KEYWORDS, "Return visit-count policy."},
    {"visit_counts", (PyCFunction)NativeMCTSGame_visit_counts, METH_NOARGS, "Return root visit counts."},
    {"root_planes", (PyCFunction)NativeMCTSGame_root_planes, METH_NOARGS, "Return root input planes."},
    {"advance_root", (PyCFunction)NativeMCTSGame_advance_root, METH_VARARGS, "Advance root by action."},
    {"is_terminal", (PyCFunction)NativeMCTSGame_is_terminal, METH_NOARGS, "Return terminal status/value."},
    {"gumbel_policy", (PyCFunction)NativeMCTSGame_gumbel_policy, METH_VARARGS | METH_KEYWORDS, "Return Gumbel completed-policy training target."},
    {"gumbel_selected_action", (PyCFunction)NativeMCTSGame_gumbel_selected_action, METH_NOARGS, "Return the Sequential-Halving-selected root action (-1 if none)."},
    {"gumbel_debug_state", (PyCFunction)NativeMCTSGame_gumbel_debug_state, METH_NOARGS, "Test-only: export root W/P/logits/noise/root_value."},
    {NULL}
};

static PyObject *native_search_batch(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *games_obj;
    PyObject *evaluator;
    int n_simulations;
    int wave_size = 16;
    int add_root_noise = 1;
    static char *kwlist[] = {
        "games", "evaluator", "n_simulations", "wave_size", "add_root_noise", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOi|ip", kwlist,
            &games_obj, &evaluator, &n_simulations, &wave_size, &add_root_noise)) {
        return NULL;
    }
    if (n_simulations < 0) {
        PyErr_SetString(PyExc_ValueError, "n_simulations must be non-negative");
        return NULL;
    }
    if (wave_size < 1) {
        PyErr_SetString(PyExc_ValueError, "wave_size must be >= 1");
        return NULL;
    }
    PyObject *seq = PySequence_Fast(games_obj, "games must be a sequence");
    if (seq == NULL) {
        return NULL;
    }
    Py_ssize_t n_games_ssize = PySequence_Fast_GET_SIZE(seq);
    if (n_games_ssize <= 0) {
        Py_DECREF(seq);
        Py_RETURN_NONE;
    }
    if (n_games_ssize > INT32_MAX) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "too many games");
        return NULL;
    }
    int n_games = (int)n_games_ssize;
    NativeMCTSGameObject **games = (NativeMCTSGameObject **)malloc(
        sizeof(NativeMCTSGameObject *) * (size_t)n_games);
    if (games == NULL) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return NULL;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (int i = 0; i < n_games; i++) {
        if (!PyObject_TypeCheck(items[i], &NativeMCTSGameType)) {
            free(games);
            Py_DECREF(seq);
            PyErr_SetString(PyExc_TypeError, "all games must be NativeMCTSGame objects");
            return NULL;
        }
        games[i] = (NativeMCTSGameObject *)items[i];
    }

    CNode **root_leaves = (CNode **)malloc(sizeof(CNode *) * (size_t)n_games);
    NativeMCTSGameObject **noise_games = (NativeMCTSGameObject **)malloc(
        sizeof(NativeMCTSGameObject *) * (size_t)n_games);
    int *sims_done = (int *)calloc((size_t)n_games, sizeof(int));
    int *wave_counts = (int *)malloc(sizeof(int) * (size_t)n_games);
    if (root_leaves == NULL || noise_games == NULL || sims_done == NULL || wave_counts == NULL) {
        free(root_leaves);
        free(noise_games);
        free(sims_done);
        free(wave_counts);
        free(games);
        Py_DECREF(seq);
        PyErr_NoMemory();
        return NULL;
    }

    int root_count = 0;
    for (int i = 0; i < n_games; i++) {
        CNode *root = &games[i]->nodes[games[i]->root];
        if (!root->expanded && !root->is_terminal) {
            root_leaves[root_count] = root;
            noise_games[root_count] = games[i];
            root_count++;
        }
    }
    if (!evaluate_and_expand(evaluator, root_leaves, root_count, noise_games, add_root_noise)) {
        free(root_leaves);
        free(noise_games);
        free(sims_done);
        free(wave_counts);
        free(games);
        Py_DECREF(seq);
        return NULL;
    }

    PendingVec pending;
    if (!pending_vec_init(&pending, n_games * wave_size > 64 ? n_games * wave_size : 64)) {
        free(root_leaves);
        free(noise_games);
        free(sims_done);
        free(wave_counts);
        free(games);
        Py_DECREF(seq);
        return NULL;
    }

    for (;;) {
        int any = 0;
        int max_slots = 0;
        for (int i = 0; i < n_games; i++) {
            int remaining = n_simulations - sims_done[i];
            int w = remaining < wave_size ? remaining : wave_size;
            if (w < 0) {
                w = 0;
            }
            wave_counts[i] = w;
            if (w > 0) {
                any = 1;
                if (w > max_slots) {
                    max_slots = w;
                }
            }
        }
        if (!any) {
            break;
        }

        pending.count = 0;
        for (int slot = 0; slot < max_slots; slot++) {
            for (int i = 0; i < n_games; i++) {
                if (slot >= wave_counts[i]) {
                    continue;
                }
                Pending p;
                if (!select_one_vloss(games[i], &p)) {
                    pending_vec_free(&pending);
                    free(root_leaves);
                    free(noise_games);
                    free(sims_done);
                    free(wave_counts);
                    free(games);
                    Py_DECREF(seq);
                    return NULL;
                }
                p.game_index = i;
                CNode *leaf = &games[i]->nodes[p.leaf_index];
                if (leaf->is_terminal) {
                    backprop_value_only(games[i], &p, leaf->terminal_value);
                } else if (!pending_vec_append(&pending, &p)) {
                    pending_vec_free(&pending);
                    free(root_leaves);
                    free(noise_games);
                    free(sims_done);
                    free(wave_counts);
                    free(games);
                    Py_DECREF(seq);
                    return NULL;
                }
            }
        }

        if (pending.count > 0) {
            CNode **leaf_nodes = (CNode **)malloc(sizeof(CNode *) * (size_t)pending.count);
            if (leaf_nodes == NULL) {
                pending_vec_free(&pending);
                free(root_leaves);
                free(noise_games);
                free(sims_done);
                free(wave_counts);
                free(games);
                Py_DECREF(seq);
                PyErr_NoMemory();
                return NULL;
            }
            for (int i = 0; i < pending.count; i++) {
                Pending *p = &pending.items[i];
                leaf_nodes[i] = &games[p->game_index]->nodes[p->leaf_index];
            }
            PyArrayObject *priors = NULL;
            PyArrayObject *values = NULL;
            if (!call_evaluator(evaluator, leaf_nodes, pending.count, &priors, &values)) {
                free(leaf_nodes);
                pending_vec_free(&pending);
                free(root_leaves);
                free(noise_games);
                free(sims_done);
                free(wave_counts);
                free(games);
                Py_DECREF(seq);
                return NULL;
            }
            float const *prior_data = (float const *)PyArray_DATA(priors);
            float const *value_data = (float const *)PyArray_DATA(values);
            for (int i = 0; i < pending.count; i++) {
                Pending *p = &pending.items[i];
                NativeMCTSGameObject *owner = games[p->game_index];
                CNode *leaf = &owner->nodes[p->leaf_index];
                if (!leaf->expanded) {
                    set_priors(leaf, prior_data + (size_t)i * N_ACTIONS);
                    leaf->expanded = 1;
                }
                backprop_value_only(owner, p, value_data[i]);
            }
            Py_DECREF(priors);
            Py_DECREF(values);
            free(leaf_nodes);
        }

        for (int i = 0; i < n_games; i++) {
            sims_done[i] += wave_counts[i];
        }
    }

    pending_vec_free(&pending);
    free(root_leaves);
    free(noise_games);
    free(sims_done);
    free(wave_counts);
    free(games);
    Py_DECREF(seq);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// Gumbel batch search. Per-game Sequential Halving lives inside the lockstep
// wave: all games run the SAME phase index together, and within a phase the
// per-game round-robin forced visits are aligned on a global slot index so the
// per-slot leaf batch stays large (preserves MPS saturation). Games with fewer
// survivors / exhausted budget simply contribute no leaf that slot — same
// pattern as wave_counts[i] in native_search_batch. Internal nodes use the
// UNCHANGED PUCT descent (select_one_vloss_gumbel forces only the root edge).
//
// Matches gomoku/gumbel.py::gumbel_search_root semantics:
//   - topk candidate sampling, SH round-robin then halve-by-score each phase,
//     final argmax of g+logits+sigma(q_hat); completed-policy target via
//     NativeMCTSGame_gumbel_policy.
//   - The per-game visits_spent is capped at n_simulations exactly like the
//     Python loop's `if visits_spent >= n_simulations: break`.
typedef struct {
    int n_phases;
    int schedule[N_ACTIONS];   // n_per for each phase
    int phase;                 // current phase index
    int phase_visits_target;   // total forced visits this phase (rounds*surv, budget-capped)
    int phase_visits_done;     // forced visits emitted this phase so far
    int sims_done;             // total forced root visits this game (<= n_simulations)
    int n_survivors;           // survivors at phase start
    unsigned char active;      // 0 once this game can no longer search
} GumbelGameSched;

static void gumbel_free_all(
    PyObject *seq, NativeMCTSGameObject **games, CNode **root_leaves,
    NativeMCTSGameObject **owner_games, GumbelGameSched *sched, PendingVec *pending
) {
    if (pending != NULL) {
        pending_vec_free(pending);
    }
    free(root_leaves);
    free(owner_games);
    free(sched);
    free(games);
    Py_XDECREF(seq);
}

static PyObject *native_gumbel_search_batch(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *games_obj;
    PyObject *evaluator;
    int n_simulations;
    int wave_size = 16;
    static char *kwlist[] = {
        "games", "evaluator", "n_simulations", "wave_size", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOi|i", kwlist,
            &games_obj, &evaluator, &n_simulations, &wave_size)) {
        return NULL;
    }
    if (n_simulations < 0) {
        PyErr_SetString(PyExc_ValueError, "n_simulations must be non-negative");
        return NULL;
    }
    if (wave_size < 1) {
        PyErr_SetString(PyExc_ValueError, "wave_size must be >= 1");
        return NULL;
    }
    PyObject *seq = PySequence_Fast(games_obj, "games must be a sequence");
    if (seq == NULL) {
        return NULL;
    }
    Py_ssize_t n_games_ssize = PySequence_Fast_GET_SIZE(seq);
    if (n_games_ssize <= 0) {
        Py_DECREF(seq);
        Py_RETURN_NONE;
    }
    if (n_games_ssize > INT32_MAX) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "too many games");
        return NULL;
    }
    int n_games = (int)n_games_ssize;
    NativeMCTSGameObject **games = (NativeMCTSGameObject **)malloc(
        sizeof(NativeMCTSGameObject *) * (size_t)n_games);
    CNode **root_leaves = (CNode **)malloc(sizeof(CNode *) * (size_t)n_games);
    NativeMCTSGameObject **owner_games = (NativeMCTSGameObject **)malloc(
        sizeof(NativeMCTSGameObject *) * (size_t)n_games);
    GumbelGameSched *sched = (GumbelGameSched *)calloc((size_t)n_games, sizeof(GumbelGameSched));
    if (games == NULL || root_leaves == NULL || owner_games == NULL || sched == NULL) {
        gumbel_free_all(seq, games, root_leaves, owner_games, sched, NULL);
        PyErr_NoMemory();
        return NULL;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (int i = 0; i < n_games; i++) {
        if (!PyObject_TypeCheck(items[i], &NativeMCTSGameType)) {
            gumbel_free_all(seq, games, root_leaves, owner_games, sched, NULL);
            PyErr_SetString(PyExc_TypeError, "all games must be NativeMCTSGame objects");
            return NULL;
        }
        games[i] = (NativeMCTSGameObject *)items[i];
    }

    // 1. Expand+evaluate all roots in ONE batched call (capture root_value +
    //    centered logits). Gumbel uses Gumbel noise, NOT Dirichlet noise.
    int root_count = 0;
    for (int i = 0; i < n_games; i++) {
        CNode *root = &games[i]->nodes[games[i]->root];
        if (!root->is_terminal) {
            root_leaves[root_count] = root;
            owner_games[root_count] = games[i];
            root_count++;
        }
    }
    if (!evaluate_and_expand_root_gumbel(evaluator, root_leaves, owner_games, root_count)) {
        gumbel_free_all(seq, games, root_leaves, owner_games, sched, NULL);
        return NULL;
    }

    // 2. Per-game Gumbel-top-k candidate sampling + SH schedule setup.
    for (int i = 0; i < n_games; i++) {
        NativeMCTSGameObject *g = games[i];
        CNode *root = &g->nodes[g->root];
        g->gumbel_selected_action = -1;
        g->gumbel_forced_action = -1;
        if (root->is_terminal) {
            sched[i].active = 0;
            continue;
        }
        gumbel_sample_topk(g, root, g->gumbel_m);
        int k = g->gumbel_n_candidates;
        if (k <= 0) {
            sched[i].active = 0;
            g->gumbel_selected_action = -1;
            continue;
        }
        if (k == 1) {
            // Single candidate: no search needed, it's the selection.
            sched[i].active = 0;
            g->gumbel_selected_action = g->gumbel_candidates[0];
            continue;
        }
        if (n_simulations <= 0) {
            // No budget: selection = argmax of g+logits+sigma(0) over candidates.
            sched[i].active = 0;
            g->gumbel_selected_action = gumbel_select_score_argmax(g, root);
            continue;
        }
        sched[i].n_phases = gumbel_sh_schedule(k, n_simulations, sched[i].schedule, N_ACTIONS);
        sched[i].phase = 0;
        sched[i].sims_done = 0;
        sched[i].active = sched[i].n_phases > 0 ? 1 : 0;
        if (!sched[i].active) {
            g->gumbel_selected_action = gumbel_select_score_argmax(g, root);
        }
    }

    PendingVec pending;
    if (!pending_vec_init(&pending, n_games * wave_size > 64 ? n_games * wave_size : 64)) {
        gumbel_free_all(seq, games, root_leaves, owner_games, sched, NULL);
        return NULL;
    }

    // 3. Phase-by-phase Sequential Halving, batched across games within a phase.
    int max_phases = 0;
    for (int i = 0; i < n_games; i++) {
        if (sched[i].active && sched[i].n_phases > max_phases) {
            max_phases = sched[i].n_phases;
        }
    }

    for (int phase = 0; phase < max_phases; phase++) {
        // Set up this phase's per-game forced-visit budget.
        for (int i = 0; i < n_games; i++) {
            GumbelGameSched *s = &sched[i];
            if (!s->active || phase >= s->n_phases || games[i]->gumbel_n_candidates <= 1) {
                s->phase_visits_target = 0;
                s->phase_visits_done = 0;
                continue;
            }
            s->n_survivors = games[i]->gumbel_n_candidates;
            int n_per = s->schedule[phase];
            int target = n_per * s->n_survivors;
            int remaining = n_simulations - s->sims_done;
            if (target > remaining) {
                target = remaining;
            }
            if (target < 0) {
                target = 0;
            }
            s->phase_visits_target = target;
            s->phase_visits_done = 0;
        }

        // Wave loop over global slot index for this phase. Each game emits its
        // scheduled forced action (round-robin over survivors) per slot.
        for (;;) {
            int any = 0;
            int max_slots = 0;
            for (int i = 0; i < n_games; i++) {
                GumbelGameSched *s = &sched[i];
                int remaining = s->phase_visits_target - s->phase_visits_done;
                int w = remaining < wave_size ? remaining : wave_size;
                if (w < 0) {
                    w = 0;
                }
                // store per-game wave count in phase_visits_done's companion:
                // reuse n_survivors? no — use a transient via remaining check.
                if (w > 0) {
                    any = 1;
                    if (w > max_slots) {
                        max_slots = w;
                    }
                }
            }
            if (!any) {
                break;
            }

            pending.count = 0;
            for (int slot = 0; slot < max_slots; slot++) {
                for (int i = 0; i < n_games; i++) {
                    GumbelGameSched *s = &sched[i];
                    int remaining = s->phase_visits_target - s->phase_visits_done;
                    int w = remaining < wave_size ? remaining : wave_size;
                    if (w < 0) {
                        w = 0;
                    }
                    if (slot >= w) {
                        continue;
                    }
                    NativeMCTSGameObject *g = games[i];
                    // Round-robin: forced action = survivors[done % n_survivors].
                    int idx = s->phase_visits_done % s->n_survivors;
                    g->gumbel_forced_action = g->gumbel_candidates[idx];
                    s->phase_visits_done += 1;
                    s->sims_done += 1;

                    Pending p;
                    if (!select_one_vloss_gumbel(g, &p)) {
                        gumbel_free_all(seq, games, root_leaves, owner_games, sched, &pending);
                        return NULL;
                    }
                    p.game_index = i;
                    CNode *leaf = &g->nodes[p.leaf_index];
                    if (leaf->is_terminal) {
                        backprop_value_only(g, &p, leaf->terminal_value);
                    } else if (!pending_vec_append(&pending, &p)) {
                        gumbel_free_all(seq, games, root_leaves, owner_games, sched, &pending);
                        return NULL;
                    }
                }
            }

            if (pending.count > 0) {
                CNode **leaf_nodes = (CNode **)malloc(sizeof(CNode *) * (size_t)pending.count);
                if (leaf_nodes == NULL) {
                    gumbel_free_all(seq, games, root_leaves, owner_games, sched, &pending);
                    PyErr_NoMemory();
                    return NULL;
                }
                for (int i = 0; i < pending.count; i++) {
                    Pending *p = &pending.items[i];
                    leaf_nodes[i] = &games[p->game_index]->nodes[p->leaf_index];
                }
                PyArrayObject *priors = NULL;
                PyArrayObject *values = NULL;
                if (!call_evaluator(evaluator, leaf_nodes, pending.count, &priors, &values)) {
                    free(leaf_nodes);
                    gumbel_free_all(seq, games, root_leaves, owner_games, sched, &pending);
                    return NULL;
                }
                float const *prior_data = (float const *)PyArray_DATA(priors);
                float const *value_data = (float const *)PyArray_DATA(values);
                for (int i = 0; i < pending.count; i++) {
                    Pending *p = &pending.items[i];
                    NativeMCTSGameObject *owner = games[p->game_index];
                    CNode *leaf = &owner->nodes[p->leaf_index];
                    if (!leaf->expanded) {
                        set_priors(leaf, prior_data + (size_t)i * N_ACTIONS);
                        leaf->expanded = 1;
                    }
                    backprop_value_only(owner, p, value_data[i]);
                }
                Py_DECREF(priors);
                Py_DECREF(values);
                free(leaf_nodes);
            }
        }

        // End of phase: halve survivors by score for each active game.
        for (int i = 0; i < n_games; i++) {
            GumbelGameSched *s = &sched[i];
            if (!s->active || phase >= s->n_phases) {
                continue;
            }
            NativeMCTSGameObject *g = games[i];
            CNode *root = &g->nodes[g->root];
            gumbel_halve_survivors(g, root);
            if (g->gumbel_n_candidates <= 1 || s->sims_done >= n_simulations
                || phase == s->n_phases - 1) {
                // No more halving possible / budget exhausted / last phase done.
                // Finalize selection now (further phases are no-ops for this game).
                s->active = 0;
                g->gumbel_selected_action = gumbel_select_score_argmax(g, root);
            }
        }
    }

    // 4. Any game still active (shouldn't normally happen) gets finalized.
    for (int i = 0; i < n_games; i++) {
        NativeMCTSGameObject *g = games[i];
        if (sched[i].active && g->gumbel_selected_action < 0) {
            CNode *root = &g->nodes[g->root];
            g->gumbel_selected_action = gumbel_select_score_argmax(g, root);
        }
    }

    gumbel_free_all(seq, games, root_leaves, owner_games, sched, &pending);
    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"search_batch", (PyCFunction)native_search_batch, METH_VARARGS | METH_KEYWORDS, "Run wave-batched native MCTS over NativeMCTSGame objects."},
    {"gumbel_search_batch", (PyCFunction)native_gumbel_search_batch, METH_VARARGS | METH_KEYWORDS, "Run wave-batched native Gumbel root + Sequential Halving over NativeMCTSGame objects."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject NativeMCTSGameType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gomoku._mcts_native.NativeMCTSGame",
    .tp_basicsize = sizeof(NativeMCTSGameObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)NativeMCTSGame_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Native arena-backed Gomoku MCTS game.",
    .tp_methods = NativeMCTSGame_methods,
    .tp_getset = NativeMCTSGame_getset,
    .tp_init = (initproc)NativeMCTSGame_init,
    .tp_new = PyType_GenericNew,
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_mcts_native",
    "Native arena-backed MCTS engine for Gomoku.",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__mcts_native(void) {
    import_array();
    init_win_masks();
    if (PyType_Ready(&NativeMCTSGameType) < 0) {
        return NULL;
    }
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }
    Py_INCREF(&NativeMCTSGameType);
    if (PyModule_AddObject(m, "NativeMCTSGame", (PyObject *)&NativeMCTSGameType) < 0) {
        Py_DECREF(&NativeMCTSGameType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
