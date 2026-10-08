"""Build a SoftHier SDK app and run it on the SoftHier GVSoC (flex_cluster target).

Standalone successor of AI_AGENT/SoftHier/dse/shdse.py so softhier-mlir needs nothing
outside this repo + the SoftHier install.

Environment
  SOFTHIER_HOME    SoftHier gvsoc checkout (default /app/install/softhier)
  SOFTHIER_CHROOT  wrapper that runs the x86-64 RISC-V toolchain on an aarch64 host
                   (default /opt/x86-ort/run; set to "" when the toolchain runs natively)
  SOFTHIER_IDEAL_HBM=1 is set for the run unless ideal_hbm=False (DRAMSys is x86-only).
  SOFTHIER_MODEL_DIR  extra gvsoc model directories (colon separated) searched first
  SOFTHIER_STOCK_MODELS=1  ignore install/models_fast and install/models_fix (stock models only)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, fields
from pathlib import Path

SHARED_HOME = Path("/app/install/softhier")   # the machine-wide install every other simulation reads
SH = Path(os.environ.get("SOFTHIER_HOME", str(SHARED_HOME)))
CHROOT = os.environ.get("SOFTHIER_CHROOT", "/opt/x86-ort/run")
SRC_GEN = SH / "soft_hier" / "flex_cluster"
PULP_GEN = SH / "pulp" / "pulp" / "chips" / "flex_cluster"
INST_GEN = SH / "install" / "generators" / "pulp" / "chips" / "flex_cluster"


def set_home(path: str | Path) -> Path:
    """Switch the SoftHier checkout used by build_sw/run_sim/apply_arch (same as SOFTHIER_HOME,
    but at run time). Architecture sweeps must point this at a private copy of the install."""
    global SH, SRC_GEN, PULP_GEN, INST_GEN
    SH = Path(path).resolve()
    SRC_GEN = SH / "soft_hier" / "flex_cluster"
    PULP_GEN = SH / "pulp" / "pulp" / "chips" / "flex_cluster"
    INST_GEN = SH / "install" / "generators" / "pulp" / "chips" / "flex_cluster"
    return SH

PERF_RE = re.compile(r"\[Performance Counter\]: Execution period is (\d+) ns")


@dataclass
class Arch:
    """The flex_cluster architecture knobs (what flex_cluster_arch.py holds)."""
    num_cluster_x: int = 4
    num_cluster_y: int = 4
    num_core_per_cluster: int = 3
    cluster_tcdm_bank_width: int = 64
    cluster_tcdm_bank_nb: int = 64
    cluster_tcdm_base: int = 0x00000000
    cluster_tcdm_size: int = 0x00100000
    cluster_tcdm_remote: int = 0x30000000
    cluster_stack_base: int = 0x10000000
    cluster_stack_size: int = 0x00020000
    cluster_zomem_base: int = 0x18000000
    cluster_zomem_size: int = 0x00020000
    cluster_reg_base: int = 0x20000000
    cluster_reg_size: int = 0x00000200
    spatz_attaced_core_list: tuple = ()
    spatz_num_vlsu_port: int = 8
    spatz_num_function_unit: int = 8
    redmule_ce_height: int = 128
    redmule_ce_width: int = 32
    redmule_ce_pipe: int = 3
    redmule_elem_size: int = 2
    redmule_queue_depth: int = 1
    redmule_reg_base: int = 0x20020000
    redmule_reg_size: int = 0x00000200
    idma_outstand_txn: int = 16
    idma_outstand_burst: int = 256
    hbm_start_base: int = 0xC0000000
    hbm_node_addr_space: int = 0x04000000
    num_node_per_ctrl: int = 1
    hbm_chan_placement: tuple = (4, 0, 0, 4)
    noc_outstanding: int = 64
    noc_link_width: int = 512
    instruction_mem_base: int = 0x80000000
    instruction_mem_size: int = 0x00010000
    soc_register_base: int = 0x90000000
    soc_register_size: int = 0x00010000
    soc_register_eoc: int = 0x90000000
    soc_register_wakeup: int = 0x90000004
    sync_base: int = 0x40000000
    sync_interleave: int = 0x00000080
    sync_special_mem: int = 0x00000040

    def to_py(self) -> str:
        lines = ["class FlexClusterArch:", "    def __init__(self):"]
        for f in fields(self):
            v = getattr(self, f.name)
            rep = f"0x{v:08x}" if isinstance(v, int) and v >= 0x1000 else repr(list(v) if isinstance(v, tuple) else v)
            lines.append(f"        self.{f.name:<34}= {rep}")
        return "\n".join(lines) + "\n"

    @property
    def n_clusters(self) -> int:
        return self.num_cluster_x * self.num_cluster_y

    @property
    def macs_per_cycle(self) -> int:
        return self.redmule_ce_height * self.redmule_ce_width


def apply_arch(arch: Arch, verbose: bool = False) -> None:
    """Write the arch into the three generator copies + regenerate the C header.
    GVSoC imports install/generators/..., the SDK includes the generated header.

    Refuses to touch the shared install (every simulation on the machine reads it at launch)
    unless SOFTHIER_ALLOW_SHARED_APPLY=1: use set_home() / SOFTHIER_HOME on a private copy."""
    if SH.resolve() == SHARED_HOME.resolve() and os.environ.get("SOFTHIER_ALLOW_SHARED_APPLY") != "1":
        raise RuntimeError(f"apply_arch on the shared install {SH} would change the architecture under every "
                           "running simulation; point SOFTHIER_HOME / set_home() at a private copy")
    (SRC_GEN / "flex_cluster_arch.py").write_text(arch.to_py())
    for dst in (PULP_GEN, INST_GEN):
        dst.mkdir(parents=True, exist_ok=True)
        for f in SRC_GEN.glob("*.py"):
            shutil.copy2(f, dst / f.name)
        shutil.rmtree(dst / "__pycache__", ignore_errors=True)
    subprocess.run([sys.executable, "soft_hier/flex_cluster_utilities/config.py",
                    "soft_hier/flex_cluster/flex_cluster_arch.py"],
                   cwd=SH, check=True, capture_output=not verbose)


def build_sw(app_dir: str | Path, arch: Arch | None = None, verbose: bool = False,
             build_dir: str | Path | None = None) -> Path:
    """Compile an SDK app dir (CMakeLists.txt + sources) into <build_dir>/softhier.elf.

    Replicates the SDK's `make sw` but in a private build directory, so several builds and
    simulations can run concurrently (default: $SOFTHIER_BUILD_DIR, else a per-process dir
    under the scratch root). The RISC-V toolchain is x86-64 only: it runs inside the chroot."""
    arch = arch or Arch()
    march = "rv32imafdv_zfh" if arch.spatz_attaced_core_list else "rv32imafd_zfh"
    bdir = Path(build_dir or os.environ.get("SOFTHIER_BUILD_DIR") or
                Path(tempfile.gettempdir()) / f"shbuild_{os.getpid()}").resolve()
    shutil.rmtree(bdir, ignore_errors=True)
    bdir.mkdir(parents=True)
    inner = (f"export PATH={SH}/third_party/toolchain/install/bin:/usr/local/bin:$PATH; "
             f"cd {bdir} && cmake -DSRC_DIR={Path(app_dir).resolve()} -DRISCV_ARCH={march} "
             f"{SH}/soft_hier/flex_cluster_sdk/ && make")
    cmd = [CHROOT, "bash", "-c", inner] if CHROOT else ["bash", "-c", inner]
    r = subprocess.run(cmd, capture_output=True, text=True)
    elf = bdir / "softhier.elf"
    if r.returncode != 0 or not elf.exists():
        raise RuntimeError("SW build failed:\n" + r.stdout[-3000:] + r.stderr[-3000:])
    if verbose:
        print(r.stdout[-500:])
    build_sw.last_elf = elf
    return elf


build_sw.last_elf = None


def run_sim(elf: Path | None = None, traces: tuple = (), timeout: int = 3600,
            ideal_hbm: bool = True, preload: Path | None = None, log: Path | None = None) -> dict:
    """Run gvsoc; returns ok/roi_ns/wall_s/stdout. ok = exited 0 and printed a ROI.

    preload: an HBM preload ELF (softhier_mlir.sim.preload.make_preload_elf); the chip's
    hbm_preloader writes it into HBM before the program's first global barrier passes.
    log: stream the simulator output to this file (for long runs) instead of only capturing it."""
    elf = elf or build_sw.last_elf or (SH / "sw_build" / "softhier.elf")
    env = dict(os.environ)
    env["SYSTEMC_HOME"] = str(SH / "third_party" / "systemc_install")
    env["LD_LIBRARY_PATH"] = f"{SH}/third_party/systemc_install/lib64:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = f"{SH}/install/python:{SH}/soft_hier/flex_cluster_utilities:" + env.get("PYTHONPATH", "")
    if ideal_hbm:
        env["SOFTHIER_IDEAL_HBM"] = "1"
    # --work-dir: gapy writes gvsoc_config.json (which names the binary) into the work dir and runs
    # from there. Without it every run shares $SOFTHIER_HOME/gvsoc_config.json and two concurrent
    # simulations (several agents, or threads) silently run each other's ELF.
    work = Path(elf).resolve().parent
    # SOFTHIER_MODEL_DIR: an extra model directory searched BEFORE install/models (the engine takes
    # the first include dir that has the .so), so a run can pin e.g. one RedMulE model build while
    # the shared install is being rebuilt by someone else. This replicates install/bin/gvsoc.
    inst = SH / "install"
    env["LD_LIBRARY_PATH"] = f"{inst}/lib:" + env["LD_LIBRARY_PATH"]
    env["PATH"] = f"{inst}/bin:" + env.get("PATH", "")
    model_dirs = [d for d in os.environ.get("SOFTHIER_MODEL_DIR", "").split(":") if d]
    # Patched Snitch models built next to the stock ones, used by default when present
    # (SOFTHIER_STOCK_MODELS=1 forces the stock models):
    #   models_fast  integer core + FP subsystem + FPU sequencer rebuilt for host speed (same simulated
    #                cycles; includes every fix below; docs/SIMULATOR_NOTES.md "Host speed of the Snitch ISS")
    #   models_fix   integer core with the fp->int `nseq` tags (docs/SIMULATOR_NOTES.md #10)
    if not os.environ.get("SOFTHIER_STOCK_MODELS"):
        for d in ("models_fast", "models_fix"):
            if (inst / d).is_dir():
                model_dirs.append(str(inst / d))
    model_dirs.append(str(inst / "models"))
    cmd = [str(inst / "bin" / "gapy"), "--platform=gvsoc", f"--target-dir={inst}/generators"] + \
          [f"--model-dir={d}" for d in model_dirs] + ["--target=pulp.chips.flex_cluster.flex_cluster",
           f"--binary={elf}", f"--work-dir={work}", "run"] + [f"--trace={t}" for t in traces]
    if preload is not None:
        cmd += ["--preload", str(Path(preload).resolve())]
    t0 = time.time()
    if log is None:
        r = subprocess.run(cmd, cwd=Path(elf).parent, env=env, capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
    else:
        with open(log, "w") as lf:
            r = subprocess.run(cmd, cwd=Path(elf).parent, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        out = Path(log).read_text()
    rois = [int(v) for v in PERF_RE.findall(out)]   # one entry per sh_timer_end()
    return {"ok": r.returncode == 0 and bool(rois), "returncode": r.returncode,
            "roi_ns": rois[0] if rois else None, "rois": rois, "wall_s": round(time.time() - t0, 1), "stdout": out}


def measure(app_dir: str | Path, arch: Arch | None = None, apply: bool = False, **run_kw) -> dict:
    """apply_arch (optional) -> build -> run."""
    if apply:
        apply_arch(arch or Arch())
    build_sw(app_dir, arch)
    return run_sim(**run_kw)
