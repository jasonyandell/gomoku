/* 11x11 build of the native state-ops extension.
 *
 * Thin shim: sets the compile-time board size and module name, then includes
 * the single shared implementation. A distinct source file (rather than
 * compiling _state_ops_native.c twice with different -D flags) keeps
 * setuptools' per-source object files from colliding between extensions.
 */
#define BOARD_SIZE 11
#define GOMOKU_STATE_OPS_MODULE_NAME _state_ops_native11

#include "_state_ops_native.c"
