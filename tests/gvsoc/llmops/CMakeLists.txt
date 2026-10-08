# SDK app: test main + the softhier-ops library (unity build). INCLUDE_DIRS must be ONE dir
# (the SDK passes it as a single -I), so test-local headers live next to main.c and the
# library headers are reached via the include dir below.
get_filename_component(SH_RT ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime ABSOLUTE)
set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c ${SH_RT}/sh_ops.c PARENT_SCOPE)
set(INCLUDE_DIRS ${SH_RT} PARENT_SCOPE)
