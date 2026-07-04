/* 11x11 build of the native MCTS extension.
 *
 * Thin shim: sets the compile-time board size and module name, then includes
 * the single shared implementation. A distinct source file (rather than
 * compiling _mcts_native.c twice with different -D flags) keeps setuptools'
 * per-source object files from colliding between extensions.
 */
#define BOARD_SIZE 11
#define GOMOKU_MCTS_MODULE_NAME _mcts_native11

#include "_mcts_native.c"
