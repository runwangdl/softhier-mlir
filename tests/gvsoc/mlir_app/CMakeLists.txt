# App dir for MLIR-generated programs: run.py writes main.c here.
get_filename_component(SH_RT ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime ABSOLUTE)
set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c ${SH_RT}/sh_ops.c PARENT_SCOPE)
set(INCLUDE_DIRS ${SH_RT} PARENT_SCOPE)
