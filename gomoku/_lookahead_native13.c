/* 13x13 build of the native lookahead extension.
 *
 * Thin shim: sets the compile-time board size and module name, then includes
 * the single shared implementation (same pattern as _state_ops_native13.c).
 */
#define BOARD_SIZE 13
#define GOMOKU_LOOKAHEAD_MODULE_NAME _lookahead_native13

#include "_lookahead_native.c"
