/* softhier-ops unity build. The SDK headers below define non-static functions and therefore
 * may only be included once per program: here. Everything else goes through sh_ops.h. */
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"
#include "flex_alloc_api.h"
#include "flex_printf.h"
#include <stdarg.h>
#include "sh_ops.h"

#ifdef SH_LIB_OPTIMIZE        /* e.g. -DSH_LIB_OPTIMIZE=\"Os\": the library (not the SDK above) for size */
#pragma GCC optimize (SH_LIB_OPTIMIZE)
#endif
#include "sh_rt.inc.c"
#include "sh_simd.inc.c"
#include "sh_gemm.inc.c"
#include "sh_l1.inc.c"
#include "sh_rowops.inc.c"
#include "sh_attention.inc.c"
#include "sh_expert.inc.c"
#include "sh_test.inc.c"
#include "sh_llm.inc.c"
#include "sh_wm.inc.c"
#include "sh_train.inc.c"
#include "sh_flow.inc.c"
