#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#define BOARD_SIZE 9
#define N_ACTIONS 81

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

static int check_plane_shape(PyArrayObject *arr) {
    if (PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "plane must have shape (9, 9)");
        return 0;
    }
    npy_intp const *dims = PyArray_DIMS(arr);
    if (dims[0] != BOARD_SIZE || dims[1] != BOARD_SIZE) {
        PyErr_SetString(PyExc_ValueError, "plane must have shape (9, 9)");
        return 0;
    }
    return 1;
}

static inline npy_bool board_get(npy_bool const *data, int plane, int row, int col) {
    return data[(plane * BOARD_SIZE + row) * BOARD_SIZE + col];
}

static inline void board_set(npy_bool *data, int plane, int row, int col, npy_bool value) {
    data[(plane * BOARD_SIZE + row) * BOARD_SIZE + col] = value;
}

static int has_five_data(npy_bool const *p) {
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c <= BOARD_SIZE - 5; c++) {
            if (p[r * BOARD_SIZE + c] &&
                p[r * BOARD_SIZE + c + 1] &&
                p[r * BOARD_SIZE + c + 2] &&
                p[r * BOARD_SIZE + c + 3] &&
                p[r * BOARD_SIZE + c + 4]) {
                return 1;
            }
        }
    }
    for (int r = 0; r <= BOARD_SIZE - 5; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (p[r * BOARD_SIZE + c] &&
                p[(r + 1) * BOARD_SIZE + c] &&
                p[(r + 2) * BOARD_SIZE + c] &&
                p[(r + 3) * BOARD_SIZE + c] &&
                p[(r + 4) * BOARD_SIZE + c]) {
                return 1;
            }
        }
    }
    for (int r = 0; r <= BOARD_SIZE - 5; r++) {
        for (int c = 0; c <= BOARD_SIZE - 5; c++) {
            if (p[r * BOARD_SIZE + c] &&
                p[(r + 1) * BOARD_SIZE + c + 1] &&
                p[(r + 2) * BOARD_SIZE + c + 2] &&
                p[(r + 3) * BOARD_SIZE + c + 3] &&
                p[(r + 4) * BOARD_SIZE + c + 4]) {
                return 1;
            }
        }
    }
    for (int r = 0; r <= BOARD_SIZE - 5; r++) {
        for (int c = 4; c < BOARD_SIZE; c++) {
            if (p[r * BOARD_SIZE + c] &&
                p[(r + 1) * BOARD_SIZE + c - 1] &&
                p[(r + 2) * BOARD_SIZE + c - 2] &&
                p[(r + 3) * BOARD_SIZE + c - 3] &&
                p[(r + 4) * BOARD_SIZE + c - 4]) {
                return 1;
            }
        }
    }
    return 0;
}

static PyObject *native_legal_mask(PyObject *self, PyObject *args) {
    PyObject *board_obj;
    if (!PyArg_ParseTuple(args, "O", &board_obj)) {
        return NULL;
    }
    PyArrayObject *board = (PyArrayObject *)PyArray_FROM_OTF(
        board_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    if (board == NULL) {
        return NULL;
    }
    if (!check_board_shape(board)) {
        Py_DECREF(board);
        return NULL;
    }

    npy_intp dims[1] = {N_ACTIONS};
    PyObject *mask_obj = PyArray_SimpleNew(1, dims, NPY_BOOL);
    if (mask_obj == NULL) {
        Py_DECREF(board);
        return NULL;
    }
    npy_bool const *b = (npy_bool const *)PyArray_DATA(board);
    npy_bool *mask = (npy_bool *)PyArray_DATA((PyArrayObject *)mask_obj);
    for (int a = 0; a < N_ACTIONS; a++) {
        int r = a / BOARD_SIZE;
        int c = a % BOARD_SIZE;
        mask[a] = !(board_get(b, 0, r, c) || board_get(b, 1, r, c));
    }
    Py_DECREF(board);
    return mask_obj;
}

static int terminal_status(PyArrayObject *board, long move_count, int *done, double *value) {
    if (!check_board_shape(board)) {
        return 0;
    }
    npy_bool const *b = (npy_bool const *)PyArray_DATA(board);
    if (has_five_data(b + N_ACTIONS)) {
        *done = 1;
        *value = -1.0;
        return 1;
    }
    if (move_count >= N_ACTIONS) {
        *done = 1;
        *value = 0.0;
        return 1;
    }
    *done = 0;
    *value = 0.0;
    return 1;
}

static PyObject *native_terminal_value(PyObject *self, PyObject *args) {
    PyObject *board_obj;
    long move_count;
    if (!PyArg_ParseTuple(args, "Ol", &board_obj, &move_count)) {
        return NULL;
    }
    PyArrayObject *board = (PyArrayObject *)PyArray_FROM_OTF(
        board_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    if (board == NULL) {
        return NULL;
    }
    int done = 0;
    double value = 0.0;
    if (!terminal_status(board, move_count, &done, &value)) {
        Py_DECREF(board);
        return NULL;
    }
    Py_DECREF(board);
    return Py_BuildValue("Nd", PyBool_FromLong(done), value);
}

static PyObject *native_init_node_status(PyObject *self, PyObject *args) {
    PyObject *board_obj;
    long move_count;
    if (!PyArg_ParseTuple(args, "Ol", &board_obj, &move_count)) {
        return NULL;
    }
    PyArrayObject *board = (PyArrayObject *)PyArray_FROM_OTF(
        board_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    if (board == NULL) {
        return NULL;
    }
    int done = 0;
    double value = 0.0;
    if (!terminal_status(board, move_count, &done, &value)) {
        Py_DECREF(board);
        return NULL;
    }
    if (done) {
        Py_DECREF(board);
        Py_INCREF(Py_None);
        return Py_BuildValue("NdO", PyBool_FromLong(1), value, Py_None);
    }

    npy_intp dims[1] = {N_ACTIONS};
    PyObject *mask_obj = PyArray_SimpleNew(1, dims, NPY_BOOL);
    if (mask_obj == NULL) {
        Py_DECREF(board);
        return NULL;
    }
    npy_bool const *b = (npy_bool const *)PyArray_DATA(board);
    npy_bool *mask = (npy_bool *)PyArray_DATA((PyArrayObject *)mask_obj);
    for (int a = 0; a < N_ACTIONS; a++) {
        int r = a / BOARD_SIZE;
        int c = a % BOARD_SIZE;
        mask[a] = !(board_get(b, 0, r, c) || board_get(b, 1, r, c));
    }
    Py_DECREF(board);
    return Py_BuildValue("NdN", PyBool_FromLong(0), 0.0, mask_obj);
}

static PyObject *native_apply_move_arrays(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *board_obj;
    PyObject *history_obj;
    int action;
    int history_ply = 8;
    static char *kwlist[] = {"board", "history", "action", "history_ply", NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOi|i", kwlist,
            &board_obj, &history_obj, &action, &history_ply)) {
        return NULL;
    }
    if (!PyTuple_Check(history_obj)) {
        PyErr_SetString(PyExc_TypeError, "history must be a tuple");
        return NULL;
    }
    PyArrayObject *board = (PyArrayObject *)PyArray_FROM_OTF(
        board_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    if (board == NULL) {
        return NULL;
    }
    if (!check_board_shape(board)) {
        Py_DECREF(board);
        return NULL;
    }
    if (action < 0 || action >= N_ACTIONS) {
        Py_DECREF(board);
        PyErr_Format(PyExc_ValueError, "illegal move %d out of range", action);
        return NULL;
    }

    int row = action / BOARD_SIZE;
    int col = action % BOARD_SIZE;
    npy_bool const *b = (npy_bool const *)PyArray_DATA(board);
    if (board_get(b, 0, row, col) || board_get(b, 1, row, col)) {
        Py_DECREF(board);
        PyErr_Format(PyExc_ValueError, "illegal move %d on occupied square", action);
        return NULL;
    }

    npy_intp board_dims[3] = {2, BOARD_SIZE, BOARD_SIZE};
    PyObject *new_board_obj = PyArray_SimpleNew(3, board_dims, NPY_BOOL);
    PyObject *snapshot_obj = PyArray_SimpleNew(3, board_dims, NPY_BOOL);
    if (new_board_obj == NULL || snapshot_obj == NULL) {
        Py_XDECREF(new_board_obj);
        Py_XDECREF(snapshot_obj);
        Py_DECREF(board);
        return NULL;
    }
    npy_bool *new_board = (npy_bool *)PyArray_DATA((PyArrayObject *)new_board_obj);
    npy_bool *snapshot = (npy_bool *)PyArray_DATA((PyArrayObject *)snapshot_obj);
    memcpy(snapshot, b, 2 * N_ACTIONS * sizeof(npy_bool));

    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            board_set(new_board, 0, r, c, board_get(b, 1, r, c));
            npy_bool mover = board_get(b, 0, r, c);
            if (r == row && c == col) {
                mover = 1;
            }
            board_set(new_board, 1, r, c, mover);
        }
    }

    Py_ssize_t hist_len = PyTuple_GET_SIZE(history_obj);
    Py_ssize_t keep = history_ply > 0 ? history_ply - 1 : 0;
    if (keep > hist_len) {
        keep = hist_len;
    }
    PyObject *new_history = PyTuple_New(1 + keep);
    if (new_history == NULL) {
        Py_DECREF(new_board_obj);
        Py_DECREF(snapshot_obj);
        Py_DECREF(board);
        return NULL;
    }
    PyTuple_SET_ITEM(new_history, 0, snapshot_obj);
    for (Py_ssize_t i = 0; i < keep; i++) {
        PyObject *item = PyTuple_GET_ITEM(history_obj, i);
        Py_INCREF(item);
        PyTuple_SET_ITEM(new_history, i + 1, item);
    }

    Py_DECREF(board);
    return Py_BuildValue("NN", new_board_obj, new_history);
}

static PyObject *native_has_five_in_a_row(PyObject *self, PyObject *args) {
    PyObject *plane_obj;
    if (!PyArg_ParseTuple(args, "O", &plane_obj)) {
        return NULL;
    }
    PyArrayObject *plane = (PyArrayObject *)PyArray_FROM_OTF(
        plane_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    if (plane == NULL) {
        return NULL;
    }
    if (!check_plane_shape(plane)) {
        Py_DECREF(plane);
        return NULL;
    }
    int has = has_five_data((npy_bool const *)PyArray_DATA(plane));
    Py_DECREF(plane);
    return PyBool_FromLong(has);
}

static PyMethodDef methods[] = {
    {"legal_mask", native_legal_mask, METH_VARARGS, "Return legal action mask."},
    {"terminal_value", native_terminal_value, METH_VARARGS, "Return terminal status and value."},
    {"init_node_status", native_init_node_status, METH_VARARGS, "Return terminal status plus legal mask."},
    {"apply_move_arrays", (PyCFunction)native_apply_move_arrays, METH_VARARGS | METH_KEYWORDS, "Apply one move to board/history arrays."},
    {"has_five_in_a_row", native_has_five_in_a_row, METH_VARARGS, "Return whether a plane has five in a row."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_state_ops_native",
    "Native hot-path state operations for Gomoku.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__state_ops_native(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
