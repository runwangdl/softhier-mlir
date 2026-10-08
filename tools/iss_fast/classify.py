#!/usr/bin/env python3
"""Classify gdb stack samples (sample.sh output) of gvsoc_launcher: for every sample take the
engine thread (the one not parked in futex/sigwait) and attribute it to one bucket by the
innermost matching frame. Prints a histogram of buckets and of the top innermost symbols."""
import re
import sys
from collections import Counter

BUCKETS = [
    ("trace sprintf (iss_trace_dump*)", r"iss_trace_dump|iss_trace_save|sprintf|vfprintf|_itoa|strchrnul|strnlen|trace_dump"),
    ("iss_insn_t copy (string/arg copies)", r"iss_insn_s::operator=|iss_insn_s::iss_insn_s|iss_insn_arg_s|basic_string|OffloadReq::operator=|BufferEntry::|OffloadRsp::operator=|_S_copy|memcpy|memmove|iss_insn_s::~"),
    ("strstr label checks", r"strstr"),
    ("sequencer", r"sequencer::"),
    ("fp subsystem offload (handle_notif/event/get_latency)", r"Iss::handle_notif|Iss::handle_event|Iss::get_latency|Iss::handle_result|Iss::handle_req|fp_offload_exec|int_offload_exec"),
    ("redmule", r"redmule|RedMule|matmul"),
    ("ISS exec/decode/lsu/prefetch", r"Exec::|Decode::|Lsu::|Prefetcher|InsnCache|iss_exec|Regfile|Csr|_exec\(|exec_insn"),
    ("vp engine / events / NoC / memory", r"vp::|ClockEngine|TimeEngine|floonoc|FlooNoc|Memory|idma|iDma|Router|router"),
]


def main(path):
    text = open(path).read()
    samples = [s for s in text.split("----\n") if s.strip()]
    buckets = Counter()
    top = Counter()
    n = 0
    for s in samples:
        threads = re.split(r"^Thread .*$", s, flags=re.M)
        for t in threads:
            frames = [ln for ln in t.splitlines() if ln.startswith("#")]
            if not frames:
                continue
            if re.search(r"futex|sigwait|sigtimedwait|pthread_cond", frames[0]):
                continue
            n += 1
            sym0 = re.sub(r"^#\d+\s+(0x[0-9a-f]+ in )?", "", frames[0]).split(" (")[0].split(" at ")[0]
            top[sym0] += 1
            stack = "\n".join(frames)
            for name, rx in BUCKETS:
                if re.search(rx, stack):
                    buckets[name] += 1
                    break
            else:
                buckets["other"] += 1
            break
    print(f"{n} engine-thread samples from {len(samples)} dumps")
    for k, v in buckets.most_common():
        print(f"  {v:4d} {100 * v / max(n, 1):5.1f}%  {k}")
    print("top innermost symbols:")
    for k, v in top.most_common(18):
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main(sys.argv[1])
