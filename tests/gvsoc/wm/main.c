/* gvsoc test: on-chip RSSM imagination (runtime/sh_wm.inc.c) on preloaded weights / states, outputs compared on the
 * host with the torch module and the fp16-floor numpy twin (tests/gvsoc/wm.py). shape.h: the RSSM dims (DETER STOCH
 * CLASSES HIDDEN UNITS OBS ACT), K H POSTERIOR RECORD KB CLUSTER NSAMPLES, the HBM offsets of the blob / inputs /
 * outputs (OFF_*) and SH_PRELOAD (the sentinel). The ROI covers the sh_wm_rssm call only (weight staging included;
 * the kernel prints its per-phase cycles of cluster 0). */
#include "sh_ops.h"
#include "shape.h"

#define S_ (STOCH * CLASSES)
#define AP_ ((ACT + 3) & ~3)

int main(void) {
    sh_init();
    const sh_wm_cfg cfg = { .deter = DETER, .stoch = STOCH, .classes = CLASSES, .hidden = HIDDEN, .units = UNITS, .obs = OBS,
                            .act = ACT, .K = K_TRAJ, .H = H_STEPS, .posterior = POSTERIOR, .record = RECORD, .kb = KB,
#ifdef OFF_DBG
                            .dbg = sh_hbm_addr(OFF_DBG)
#else
                            .dbg = 0
#endif
                          };
    const uint64_t blob = sh_hbm_addr(OFF_BLOB), h0 = sh_hbm_addr(OFF_H0), z0 = sh_hbm_addr(OFF_Z0), a0 = sh_hbm_addr(OFF_A0);
    const uint64_t obs = sh_hbm_addr(OFF_OBS), hs = sh_hbm_addr(OFF_HS), zs = sh_hbm_addr(OFF_ZS), as = sh_hbm_addr(OFF_AS);
    const int lead = (sh_cluster_id() == 0 && sh_is_first_core());
    sh_preload_wait(sh_hbm_addr(SH_PRELOAD));
    if (lead) sh_timer_start();
    const int rc = sh_wm_rssm(&cfg, blob, h0, z0, a0, obs, hs, zs, as, CLUSTER);
    sh_barrier_global();
    if (lead) {
        sh_timer_end();
        const uint32_t rows = (RECORD ? H_STEPS : 1) * K_TRAJ;
        sh_test_dump_samples(hs, rows, DETER, DETER, 101, NSAMPLES, "HS");
        sh_test_dump_samples(zs, rows, S_, S_, 102, NSAMPLES, "ZS");
        sh_test_dump_samples(as, rows, AP_, AP_, 103, NSAMPLES, "AS");
#ifdef OFF_DBG
        { static const uint32_t w[10] = { DBG_W0, DBG_W1, DBG_W2, DBG_W3, DBG_W4, DBG_W5, DBG_W6, DBG_W7, DBG_W8, DBG_W9 };
          for (uint32_t i = 0; i < 10; ++i) sh_test_dump_samples_idx(sh_hbm_addr(OFF_DBG) + i * 0x10000u, DBG_M, w[i], w[i], 200 + i, 128, "D", i); }
#endif
        sh_printf(rc ? "WM_FAIL rc=%d\n" : "WM_DONE rc=%d\n", rc);
    }
    sh_barrier_global();
    sh_eoc(0);
    return 0;
}
