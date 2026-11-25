# softhier-mlir + SoftHier GVSoC (flex_cluster), one image.
#
#   compiler      softhier-mlir (xDSL dialect, softhier-opt / softhier-translate)   /app/softhier-mlir
#   simulator     gvsoc/gvsoc @ bowwang-dev/softhier-deeploy + the model fixes      /app/install/softhier
#                 (built the way pulp-platform/Deeploy's container does: `make softhier`)
#   RISC-V gcc    prebuilt x86-64 toolchain (pulp gcc + rv32 V/Zfh gcc)             .../third_party/toolchain
#
# The image builds natively on arm64 (Apple Silicon) and on amd64. The RISC-V toolchain only exists
# as x86-64 binaries: on arm64 the x86-64 loader + glibc are copied in from an amd64 stage, so the
# compiler runs through Docker's binfmt emulation while gvsoc itself stays native. DRAMSys is an
# x86-64 .so, so the HBM is gvsoc's ideal memory on every host (SOFTHIER_IDEAL_HBM=1, what all
# numbers in the repo were measured with).

ARG UBUNTU=22.04

# ---- x86-64 userland the prebuilt RISC-V toolchain needs (only used on non-x86 hosts) -----------
FROM --platform=linux/amd64 ubuntu:${UBUNTU} AS x86libs
RUN apt-get update && apt-get install -y --no-install-recommends zlib1g libstdc++6 libexpat1 libncurses5 \
    && rm -rf /var/lib/apt/lists/*

# ---- main image ---------------------------------------------------------------------------------
FROM ubuntu:${UBUNTU}
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates curl wget rsync patch xz-utils bzip2 file less vim-tiny \
      python3 python3-pip python3-venv python3-dev \
      autoconf automake texinfo libtool pkg-config ccache \
      libsdl2-dev libsdl2-ttf-dev libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,from=x86libs,source=/usr/lib,target=/x86/usr/lib \
    if [ "$(uname -m)" != "x86_64" ]; then \
      cp -a /x86/usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu && \
      mkdir -p /lib64 && ln -sf /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2 ; \
    fi

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN python3 -m venv /opt/venv && pip install --no-cache-dir --upgrade pip wheel "cmake>=3.28,<4"

# ---- SoftHier GVSoC source ----------------------------------------------------------------------
ENV SOFTHIER_HOME=/app/install/softhier
RUN git clone --depth 1 -b bowwang-dev/softhier-deeploy https://github.com/gvsoc/gvsoc.git $SOFTHIER_HOME \
    && cd $SOFTHIER_HOME && git submodule update --init --recursive --depth 1 \
    && git log -1 --format='gvsoc %h %cd' && git submodule status
WORKDIR $SOFTHIER_HOME
# Same runtime swap as pulp-platform/Deeploy's `make softhier` (Makefile, target ${TOOLCHAIN_DIR}/softhier):
# the Deeploy linker script + flex_alloc/flex_runtime headers. softhier-mlir's runtime/ is written against these.
RUN cd soft_hier/flex_cluster_sdk/runtime \
    && rm flex_memory.ld include/flex_alloc.h include/flex_runtime.h \
    && mv flex_memory_deeploy.ld flex_memory.ld && cp deeploy_include/* include/
RUN pip install --no-cache-dir -r requirements.txt -r core/requirements.txt -r gapy/requirements.txt numpy

# RISC-V toolchain (x86-64 prebuilt, rv32 + V + Zfh). Only the C compiler is kept: the C++ / LTO
# front ends and gdb are 1.9 GB; the older pulp gcc (1.1 GB) that the SDK Makefile also downloads is unused.
RUN mkdir -p third_party/toolchain && cd third_party/toolchain \
    && wget -q https://github.com/husterZC/gun_toolchain/releases/download/v2.0.0/toolchain.tar.xz \
    && tar -xf toolchain.tar.xz && rm toolchain.tar.xz \
    && G=install/libexec/gcc/riscv32-unknown-elf/14.2.0 \
    && rm -rf $G/cc1plus $G/lto1 $G/g++-mapper-server $G/plugin/libcc1* \
         install/bin/riscv32-unknown-elf-lto-dump install/bin/riscv32-unknown-elf-gdb* \
         install/bin/riscv32-unknown-elf-c++ install/bin/riscv32-unknown-elf-g++ \
    && install/bin/riscv32-unknown-elf-gcc --version | head -1

# SystemC (the dramsys component is compiled against it even when the ideal HBM is used)
ENV SYSTEMC_HOME=$SOFTHIER_HOME/third_party/systemc_install
RUN make build-systemc CMAKE=cmake && rm -rf third_party/systemc

# SoftHier's own patches to gvsoc-core / gvsoc-pulp, dramsys configs
RUN make drmasys_apply_patch build-configs \
    && mkdir -p third_party/DRAMSys && cp soft_hier/libDRAMSys_Simulator.so third_party/DRAMSys/

# ---- model fixes (docs/SIMULATOR_NOTES.md of softhier-mlir; AI_AGENT/SoftHier/dse/patches) ------
COPY patches /app/patches
RUN set -e; P=/app/patches; \
    patch -p0 < $P/flex_cluster_ideal_hbm.patch; \
    patch -p4 < $P/gvsoc_redmule_fp16_rne.patch; \
    patch -p1 < $P/gvsoc_redmule_neon.patch; \
    (cd core/models/cpu/iss && patch -p0 < $P/gvsoc_snitch_fp_ss_pending.patch); \
    patch -p0 < $P/gvsoc_zfh_flh_fsh.patch; \
    patch -p0 < $P/gvsoc_snitch_fp_int_nseq.patch; \
    patch -p0 < $P/gvsoc_iss_host_speed.patch

# ---- build the flex_cluster platform ------------------------------------------------------------
ENV LD_LIBRARY_PATH=$SOFTHIER_HOME/third_party/systemc_install/lib64:$SOFTHIER_HOME/third_party/DRAMSys
ARG JOBS=6
# = Deeploy's `make hw-deeploy` (arch_deeploy.py), minus the `checkout` step that would reset the submodules
RUN config_file=examples/SoftHier/config/arch_deeploy.py make config \
    && make TARGETS=pulp.chips.flex_cluster.flex_cluster build CMAKE_FLAGS="-j ${JOBS}" \
    && rm -rf build/core/CMakeFiles build/pulp/CMakeFiles

# ---- softhier-mlir ------------------------------------------------------------------------------
ARG MLIR_REF=main
RUN git clone https://github.com/runwangdl/softhier-mlir.git /app/softhier-mlir \
    && cd /app/softhier-mlir && git checkout -q ${MLIR_REF} \
    && pip install --no-cache-dir -e ".[dev]"

# The architecture softhier-mlir assumes (softhier_mlir.sim.gvsoc.Arch defaults: 4x4 clusters, HBM on the
# west + south edges, 64 MiB per HBM node). arch_deeploy.py has HBM on the west edge only: programs that
# touch hbm_south segfault gvsoc in hbm_ctrl. Python generators + C header only, no model rebuild.
RUN SOFTHIER_ALLOW_SHARED_APPLY=1 python -c "from softhier_mlir.sim.gvsoc import Arch, apply_arch; apply_arch(Arch())"

# toolchain runs directly (natively on amd64, through binfmt on arm64): no chroot wrapper
ENV SOFTHIER_CHROOT="" \
    SOFTHIER_IDEAL_HBM=1 \
    PATH=$SOFTHIER_HOME/install/bin:$SOFTHIER_HOME/third_party/toolchain/install/bin:$PATH \
    PYTHONPATH=$SOFTHIER_HOME/install/python:$SOFTHIER_HOME/soft_hier/flex_cluster_utilities

LABEL org.opencontainers.image.source=https://github.com/runwangdl/softhier-mlir \
      org.opencontainers.image.description="softhier-mlir + SoftHier GVSoC (flex_cluster) + RISC-V toolchain" \
      org.opencontainers.image.licenses=Apache-2.0
COPY smoke.sh /usr/local/bin/smoke
WORKDIR /app/softhier-mlir
CMD ["bash"]
