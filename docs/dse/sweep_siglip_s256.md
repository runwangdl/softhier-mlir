Workload `siglip_S256_L1`: 55 ops, 1.91 GMAC. Model = analytic estimate; composed = per-shape kernel simulations x counts through the same timeline (not an end-to-end measurement).

| noc_link_width | redmule_ce | mesh | model us | composed us | err % | softmax (model/composed us) | add_bias (model/composed us) | layernorm (model/composed us) | gelu (model/composed us) | add (model/composed us) | transpose (model/composed us) | gemm (model/composed us) | barrier (model/composed us) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 64x64 | 4x4 | 49527.1 | 52827.5 | -6.2 | 21837.9 / 21828.3 | 10197.5 / 13535.8 | 7477.1 / 7466.2 | 7269.9 / 7266.2 | 2268.4 / 2264.7 | 206.0 / 206.9 | 270.1 / 259.2 | 0.2 / 0.0 |
| 1024 | 128x32 | 4x4 | 49527.2 | 52827.9 | -6.2 | 21837.9 / 21828.3 | 10197.5 / 13535.8 | 7477.1 / 7466.2 | 7269.9 / 7266.2 | 2268.4 / 2264.7 | 206.0 / 206.9 | 270.2 / 259.7 | 0.2 / 0.0 |
| 512 | 128x32 | 4x4 | 49810.5 | 53042.1 | -6.1 | 21854.4 / 21829.4 | 10240.5 / 13563.6 | 7486.3 / 7475.3 | 7288.4 / 7284.6 | 2282.3 / 2277.3 | 208.7 / 208.3 | 449.6 / 403.5 | 0.2 / 0.0 |
| 512 | 64x64 | 4x4 | 49811.6 | 53043.2 | -6.1 | 21854.4 / 21829.4 | 10240.5 / 13563.6 | 7486.3 / 7475.3 | 7288.4 / 7284.6 | 2282.3 / 2277.3 | 208.7 / 208.3 | 450.7 / 404.6 | 0.2 / 0.0 |
| 256 | 128x32 | 4x4 | 50402.2 | 53509.1 | -5.8 | 21887.4 / 21831.4 | 10326.5 / 13619.7 | 7504.9 / 7492.8 | 7325.5 / 7321.4 | 2310.1 / 2303.8 | 218.0 / 213.4 | 829.6 / 726.5 | 0.2 / 0.0 |
| 256 | 64x64 | 4x4 | 50403.3 | 53510.2 | -5.8 | 21887.4 / 21831.4 | 10326.5 / 13619.7 | 7504.9 / 7492.8 | 7325.5 / 7321.4 | 2310.1 / 2303.8 | 218.0 / 213.4 | 830.7 / 727.6 | 0.2 / 0.0 |
| 1024 | 128x32 | 1x1 | 700242.7 | 673502.9 | +4.0 | 261936.2 / 245401.5 | 162466.9 / 162179.9 | 119479.2 / 112666.9 | 116040.9 / 115163.9 | 36070.3 / 36039.1 | 3258.0 / 1135.7 | 990.8 / 915.9 | 0.2 / 0.0 |
| 1024 | 64x64 | 1x1 | 700253.4 | 673510.2 | +4.0 | 261936.2 / 245401.5 | 162466.9 / 162179.9 | 119479.2 / 112666.9 | 116040.9 / 115163.9 | 36070.3 / 36039.1 | 3258.0 / 1135.7 | 1001.6 / 923.1 | 0.2 / 0.0 |
| 512 | 128x32 | 1x1 | 700423.3 | 673676.6 | +4.0 | 261948.5 / 245413.8 | 162510.3 / 162221.3 | 119486.4 / 112673.2 | 116065.5 / 115188.5 | 36083.6 / 36051.5 | 3261.1 / 1138.7 | 1067.6 / 989.6 | 0.2 / 0.0 |
| 512 | 64x64 | 1x1 | 700434.0 | 673685.8 | +4.0 | 261948.5 / 245413.8 | 162510.3 / 162221.3 | 119486.4 / 112673.2 | 116065.5 / 115188.5 | 36083.6 / 36051.5 | 3261.1 / 1138.7 | 1078.4 / 998.8 | 0.2 / 0.0 |
| 256 | 128x32 | 1x1 | 701142.4 | 674361.0 | +4.0 | 261973.1 / 245438.5 | 162622.6 / 162329.3 | 119511.0 / 112693.2 | 116114.6 / 115237.7 | 36120.5 / 36084.8 | 3267.2 / 1145.0 | 1533.1 / 1432.5 | 0.2 / 0.0 |
| 256 | 64x64 | 1x1 | 701158.3 | 674376.7 | +4.0 | 261973.1 / 245438.5 | 162622.6 / 162329.3 | 119511.0 / 112693.2 | 116114.6 / 115237.7 | 36120.5 / 36084.8 | 3267.2 / 1145.0 | 1549.0 / 1448.3 | 0.2 / 0.0 |

**{'noc_link_width': 1024, 'redmule_ce': '64x64', 'mesh': '4x4'}**

| kernel | count | model cyc | sim cyc | err % | model note |
|---|---|---|---|---|---|
| layernorm 256x768 all | 2 | 3,738,535 | 3733108 | +0.1 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x768x768 tile 256x256x256 pipe=1 acc=0 all | 4 | 25,270 | 26077 | -3.1 | dma: 1 tiles/cluster, 3 clusters streaming, tile 4694 vs load 4634 |
| add_bias 256x768 all | 5 | 1,133,417 | 1131128 | +0.2 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| transpose 256x768 all | 1 | 205,980 | 206943 | -0.5 | scalar: 3 blocks/cluster |
| gemm 256x256x64 tile 256x256x64 pipe=1 acc=0 c0 | 12 | 8,778 | 9691 | -9.4 | dma: 1 tiles/cluster, 1 clusters streaming, tile 1622 vs load 905 |
| softmax 256x256 c0 | 12 | 21,828,019 | 21828338 | -0.0 | scalar: 1 blocks, 256 rows on the critical cluster (block 256 rows) |
| gemm 256x64x256 tile 256x64x256 pipe=1 acc=0 c0 | 12 | 12,510 | 13339 | -6.2 | dma: 1 tiles/cluster, 1 clusters streaming, tile 4694 vs load 1696 |
| add 256x768 all | 2 | 1,134,211 | 1132350 | +0.2 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x3072x768 tile 256x256x256 pipe=1 acc=0 all | 1 | 64,644 | 63591 | +1.7 | dma: 1 tiles/cluster, 12 clusters streaming, tile 4694 vs load 16924 |
| add_bias 256x3072 all | 1 | 4,530,431 | 7880172 | -42.5 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gelu 256x3072 all | 1 | 7,269,900 | 7266207 | +0.1 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x768x3072 tile 256x256x256 pipe=1 acc=0 all | 1 | 67,786 | 68307 | -0.8 | dma: 1 tiles/cluster, 3 clusters streaming, tile 4694 vs load 4634 |

**{'noc_link_width': 1024, 'redmule_ce': '128x32', 'mesh': '4x4'}**

| kernel | count | model cyc | sim cyc | err % | model note |
|---|---|---|---|---|---|
| layernorm 256x768 all | 2 | 3,738,535 | 3733108 | +0.1 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x768x768 tile 256x256x256 pipe=1 acc=0 all | 4 | 25,462 | 26305 | -3.2 | dma: 1 tiles/cluster, 3 clusters streaming, tile 4758 vs load 4634 |
| add_bias 256x768 all | 5 | 1,133,417 | 1131121 | +0.2 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| transpose 256x768 all | 1 | 205,980 | 206939 | -0.5 | scalar: 3 blocks/cluster |
| gemm 256x256x64 tile 256x256x64 pipe=1 acc=0 c0 | 12 | 9,290 | 10203 | -9.0 | dma: 1 tiles/cluster, 1 clusters streaming, tile 2134 vs load 905 |
| softmax 256x256 c0 | 12 | 21,828,019 | 21828338 | -0.0 | scalar: 1 blocks, 256 rows on the critical cluster (block 256 rows) |
| gemm 256x64x256 tile 256x64x256 pipe=1 acc=0 c0 | 12 | 10,526 | 11355 | -7.3 | dma: 1 tiles/cluster, 1 clusters streaming, tile 2710 vs load 1696 |
| add 256x768 all | 2 | 1,134,211 | 1132374 | +0.2 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x3072x768 tile 256x256x256 pipe=1 acc=0 all | 1 | 64,708 | 63649 | +1.7 | dma: 1 tiles/cluster, 12 clusters streaming, tile 4758 vs load 16924 |
| add_bias 256x3072 all | 1 | 4,530,431 | 7880180 | -42.5 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gelu 256x3072 all | 1 | 7,269,900 | 7266207 | +0.1 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x768x3072 tile 256x256x256 pipe=1 acc=0 all | 1 | 68,554 | 69279 | -1.0 | dma: 1 tiles/cluster, 3 clusters streaming, tile 4758 vs load 4634 |

**{'noc_link_width': 512, 'redmule_ce': '128x32', 'mesh': '4x4'}**

| kernel | count | model cyc | sim cyc | err % | model note |
|---|---|---|---|---|---|
| layernorm 256x768 all | 2 | 3,743,167 | 3737659 | +0.1 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x768x768 tile 256x256x256 pipe=1 acc=0 all | 4 | 37,869 | 37198 | +1.8 | dma: 1 tiles/cluster, 3 clusters streaming, tile 4758 vs load 8852 |
| add_bias 256x768 all | 5 | 1,138,193 | 1135518 | +0.2 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| transpose 256x768 all | 1 | 208,705 | 208296 | +0.2 | scalar: 3 blocks/cluster |
| gemm 256x256x64 tile 256x256x64 pipe=1 acc=0 c0 | 12 | 9,802 | 10719 | -8.6 | dma: 1 tiles/cluster, 1 clusters streaming, tile 2134 vs load 1417 |
| softmax 256x256 c0 | 12 | 21,829,043 | 21829366 | -0.0 | scalar: 1 blocks, 256 rows on the critical cluster (block 256 rows) |
| gemm 256x64x256 tile 256x64x256 pipe=1 acc=0 c0 | 12 | 11,806 | 12635 | -6.6 | dma: 1 tiles/cluster, 1 clusters streaming, tile 2710 vs load 2976 |
| add 256x768 all | 2 | 1,141,159 | 1138668 | +0.2 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x3072x768 tile 256x256x256 pipe=1 acc=0 all | 1 | 122,486 | 120435 | +1.7 | dma: 1 tiles/cluster, 12 clusters streaming, tile 4758 vs load 33432 |
| add_bias 256x3072 all | 1 | 4,549,538 | 7886056 | -42.3 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gelu 256x3072 all | 1 | 7,288,428 | 7284625 | +0.1 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x768x3072 tile 256x256x256 pipe=1 acc=0 all | 1 | 117,810 | 110933 | +6.2 | dma: 1 tiles/cluster, 3 clusters streaming, tile 4758 vs load 8852 |

**{'noc_link_width': 512, 'redmule_ce': '64x64', 'mesh': '4x4'}**

| kernel | count | model cyc | sim cyc | err % | model note |
|---|---|---|---|---|---|
| layernorm 256x768 all | 2 | 3,743,167 | 3737659 | +0.1 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x768x768 tile 256x256x256 pipe=1 acc=0 all | 4 | 37,805 | 37138 | +1.8 | dma: 1 tiles/cluster, 3 clusters streaming, tile 4694 vs load 8852 |
| add_bias 256x768 all | 5 | 1,138,193 | 1135518 | +0.2 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| transpose 256x768 all | 1 | 208,705 | 208296 | +0.2 | scalar: 3 blocks/cluster |
| gemm 256x256x64 tile 256x256x64 pipe=1 acc=0 c0 | 12 | 9,290 | 10207 | -9.0 | dma: 1 tiles/cluster, 1 clusters streaming, tile 1622 vs load 1417 |
| softmax 256x256 c0 | 12 | 21,829,043 | 21829366 | -0.0 | scalar: 1 blocks, 256 rows on the critical cluster (block 256 rows) |
| gemm 256x64x256 tile 256x64x256 pipe=1 acc=0 c0 | 12 | 13,790 | 14619 | -5.7 | dma: 1 tiles/cluster, 1 clusters streaming, tile 4694 vs load 2976 |
| add 256x768 all | 2 | 1,141,159 | 1138668 | +0.2 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x3072x768 tile 256x256x256 pipe=1 acc=0 all | 1 | 122,422 | 120373 | +1.7 | dma: 1 tiles/cluster, 12 clusters streaming, tile 4694 vs load 33432 |
| add_bias 256x3072 all | 1 | 4,549,538 | 7886038 | -42.3 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gelu 256x3072 all | 1 | 7,288,428 | 7284623 | +0.1 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x768x3072 tile 256x256x256 pipe=1 acc=0 all | 1 | 117,746 | 110871 | +6.2 | dma: 1 tiles/cluster, 3 clusters streaming, tile 4694 vs load 8852 |

**{'noc_link_width': 256, 'redmule_ce': '128x32', 'mesh': '4x4'}**

| kernel | count | model cyc | sim cyc | err % | model note |
|---|---|---|---|---|---|
| layernorm 256x768 all | 2 | 3,752,431 | 3746424 | +0.2 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x768x768 tile 256x256x256 pipe=1 acc=0 all | 4 | 65,982 | 63606 | +3.7 | dma: 1 tiles/cluster, 3 clusters streaming, tile 4758 vs load 17288 |
| add_bias 256x768 all | 5 | 1,147,747 | 1144430 | +0.3 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| transpose 256x768 all | 1 | 217,969 | 213383 | +2.1 | scalar: 3 blocks/cluster |
| gemm 256x256x64 tile 256x256x64 pipe=1 acc=0 c0 | 12 | 10,826 | 11751 | -7.9 | dma: 1 tiles/cluster, 1 clusters streaming, tile 2134 vs load 2441 |
| softmax 256x256 c0 | 12 | 21,831,091 | 21831422 | -0.0 | scalar: 1 blocks, 256 rows on the critical cluster (block 256 rows) |
| gemm 256x64x256 tile 256x64x256 pipe=1 acc=0 c0 | 12 | 14,366 | 15195 | -5.5 | dma: 1 tiles/cluster, 1 clusters streaming, tile 2710 vs load 5536 |
| add 256x768 all | 2 | 1,155,055 | 1151894 | +0.3 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x3072x768 tile 256x256x256 pipe=1 acc=0 all | 1 | 238,042 | 234527 | +1.5 | dma: 1 tiles/cluster, 12 clusters streaming, tile 4758 vs load 66448 |
| add_bias 256x3072 all | 1 | 4,587,752 | 7897584 | -41.9 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gelu 256x3072 all | 1 | 7,325,484 | 7321437 | +0.1 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x768x3072 tile 256x256x256 pipe=1 acc=0 all | 1 | 221,847 | 210623 | +5.3 | dma: 1 tiles/cluster, 3 clusters streaming, tile 4758 vs load 17288 |

**{'noc_link_width': 256, 'redmule_ce': '64x64', 'mesh': '4x4'}**

| kernel | count | model cyc | sim cyc | err % | model note |
|---|---|---|---|---|---|
| layernorm 256x768 all | 2 | 3,752,431 | 3746424 | +0.2 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x768x768 tile 256x256x256 pipe=1 acc=0 all | 4 | 65,918 | 63532 | +3.8 | dma: 1 tiles/cluster, 3 clusters streaming, tile 4694 vs load 17288 |
| add_bias 256x768 all | 5 | 1,147,747 | 1144430 | +0.3 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| transpose 256x768 all | 1 | 217,969 | 213383 | +2.1 | scalar: 3 blocks/cluster |
| gemm 256x256x64 tile 256x256x64 pipe=1 acc=0 c0 | 12 | 10,314 | 11239 | -8.2 | dma: 1 tiles/cluster, 1 clusters streaming, tile 1622 vs load 2441 |
| softmax 256x256 c0 | 12 | 21,831,091 | 21831422 | -0.0 | scalar: 1 blocks, 256 rows on the critical cluster (block 256 rows) |
| gemm 256x64x256 tile 256x64x256 pipe=1 acc=0 c0 | 12 | 16,350 | 17179 | -4.8 | dma: 1 tiles/cluster, 1 clusters streaming, tile 4694 vs load 5536 |
| add 256x768 all | 2 | 1,155,055 | 1151894 | +0.3 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x3072x768 tile 256x256x256 pipe=1 acc=0 all | 1 | 237,978 | 234473 | +1.5 | dma: 1 tiles/cluster, 12 clusters streaming, tile 4694 vs load 66448 |
| add_bias 256x3072 all | 1 | 4,587,752 | 7897584 | -41.9 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gelu 256x3072 all | 1 | 7,325,484 | 7321431 | +0.1 | scalar: 1 blocks, 16 rows on the critical cluster (block 16 rows) |
| gemm 256x768x3072 tile 256x256x256 pipe=1 acc=0 all | 1 | 221,783 | 210553 | +5.3 | dma: 1 tiles/cluster, 3 clusters streaming, tile 4694 vs load 17288 |

**{'noc_link_width': 1024, 'redmule_ce': '128x32', 'mesh': '1x1'}**

| kernel | count | model cyc | sim cyc | err % | model note |
|---|---|---|---|---|---|
| layernorm 256x768 all | 2 | 59,739,625 | 56333472 | +6.0 | scalar: 4 blocks, 256 rows on the critical cluster (block 85 rows) |
| gemm 256x768x768 tile 256x256x256 pipe=1 acc=0 all | 4 | 69,389 | 64398 | +7.8 | compute: 3 tiles/cluster, 1 clusters streaming, tile 4758 vs load 2464 |
| add_bias 256x768 all | 5 | 18,053,336 | 18033152 | +0.1 | scalar: 4 blocks, 256 rows on the critical cluster (block 85 rows) |
| transpose 256x768 all | 1 | 3,258,028 | 1135661 | +186.9 | scalar: 48 blocks/cluster |
| gemm 256x256x64 tile 256x256x64 pipe=1 acc=0 c0 | 12 | 9,290 | 8164 | +13.8 | dma: 1 tiles/cluster, 1 clusters streaming, tile 2134 vs load 905 |
| softmax 256x256 c0 | 12 | 21,828,019 | 20450123 | +6.7 | scalar: 1 blocks, 256 rows on the critical cluster (block 256 rows) |
| gemm 256x64x256 tile 256x64x256 pipe=1 acc=0 c0 | 12 | 10,526 | 9314 | +13.0 | dma: 1 tiles/cluster, 1 clusters streaming, tile 2710 vs load 1696 |
| add 256x768 all | 2 | 18,035,160 | 18019539 | +0.1 | scalar: 5 blocks, 256 rows on the critical cluster (block 56 rows) |
| gemm 256x3072x768 tile 256x256x256 pipe=1 acc=0 all | 1 | 276,826 | 254163 | +8.9 | compute: 12 tiles/cluster, 1 clusters streaming, tile 4758 vs load 2464 |
| add_bias 256x3072 all | 1 | 72,200,218 | 72014174 | +0.3 | scalar: 13 blocks, 256 rows on the critical cluster (block 21 rows) |
| gelu 256x3072 all | 1 | 116,040,900 | 115163934 | +0.8 | scalar: 13 blocks, 256 rows on the critical cluster (block 21 rows) |
| gemm 256x768x3072 tile 256x256x256 pipe=1 acc=0 all | 1 | 198,665 | 194388 | +2.2 | compute: 3 tiles/cluster, 1 clusters streaming, tile 4758 vs load 2464 |

**{'noc_link_width': 1024, 'redmule_ce': '64x64', 'mesh': '1x1'}**

| kernel | count | model cyc | sim cyc | err % | model note |
|---|---|---|---|---|---|
| layernorm 256x768 all | 2 | 59,739,625 | 56333472 | +6.0 | scalar: 4 blocks, 256 rows on the critical cluster (block 85 rows) |
| gemm 256x768x768 tile 256x256x256 pipe=1 acc=0 all | 4 | 68,813 | 63564 | +8.3 | compute: 3 tiles/cluster, 1 clusters streaming, tile 4694 vs load 2464 |
| add_bias 256x768 all | 5 | 18,053,336 | 18033152 | +0.1 | scalar: 4 blocks, 256 rows on the critical cluster (block 85 rows) |
| transpose 256x768 all | 1 | 3,258,028 | 1135661 | +186.9 | scalar: 48 blocks/cluster |
| gemm 256x256x64 tile 256x256x64 pipe=1 acc=0 c0 | 12 | 8,778 | 7652 | +14.7 | dma: 1 tiles/cluster, 1 clusters streaming, tile 1622 vs load 905 |
| softmax 256x256 c0 | 12 | 21,828,019 | 20450123 | +6.7 | scalar: 1 blocks, 256 rows on the critical cluster (block 256 rows) |
| gemm 256x64x256 tile 256x64x256 pipe=1 acc=0 c0 | 12 | 12,510 | 11298 | +10.7 | dma: 1 tiles/cluster, 1 clusters streaming, tile 4694 vs load 1696 |
| add 256x768 all | 2 | 18,035,160 | 18019539 | +0.1 | scalar: 5 blocks, 256 rows on the critical cluster (block 56 rows) |
| gemm 256x3072x768 tile 256x256x256 pipe=1 acc=0 all | 1 | 274,522 | 250827 | +9.4 | compute: 12 tiles/cluster, 1 clusters streaming, tile 4694 vs load 2464 |
| add_bias 256x3072 all | 1 | 72,200,218 | 72014174 | +0.3 | scalar: 13 blocks, 256 rows on the critical cluster (block 21 rows) |
| gelu 256x3072 all | 1 | 116,040,900 | 115163934 | +0.8 | scalar: 13 blocks, 256 rows on the critical cluster (block 21 rows) |
| gemm 256x768x3072 tile 256x256x256 pipe=1 acc=0 all | 1 | 196,361 | 190665 | +3.0 | compute: 3 tiles/cluster, 1 clusters streaming, tile 4694 vs load 2464 |

**{'noc_link_width': 512, 'redmule_ce': '128x32', 'mesh': '1x1'}**

| kernel | count | model cyc | sim cyc | err % | model note |
|---|---|---|---|---|---|
| layernorm 256x768 all | 2 | 59,743,210 | 56336592 | +6.0 | scalar: 4 blocks, 256 rows on the critical cluster (block 85 rows) |
| gemm 256x768x768 tile 256x256x256 pipe=1 acc=0 all | 4 | 75,533 | 70317 | +7.4 | dma: 3 tiles/cluster, 1 clusters streaming, tile 4758 vs load 4512 |
| add_bias 256x768 all | 5 | 18,056,969 | 18036324 | +0.1 | scalar: 4 blocks, 256 rows on the critical cluster (block 85 rows) |
| transpose 256x768 all | 1 | 3,261,100 | 1138733 | +186.4 | scalar: 48 blocks/cluster |
| gemm 256x256x64 tile 256x256x64 pipe=1 acc=0 c0 | 12 | 9,802 | 8692 | +12.8 | dma: 1 tiles/cluster, 1 clusters streaming, tile 2134 vs load 1417 |
| softmax 256x256 c0 | 12 | 21,829,043 | 20451151 | +6.7 | scalar: 1 blocks, 256 rows on the critical cluster (block 256 rows) |
| gemm 256x64x256 tile 256x64x256 pipe=1 acc=0 c0 | 12 | 11,806 | 10594 | +11.4 | dma: 1 tiles/cluster, 1 clusters streaming, tile 2710 vs load 2976 |
| add 256x768 all | 2 | 18,041,817 | 18025743 | +0.1 | scalar: 5 blocks, 256 rows on the critical cluster (block 56 rows) |
| gemm 256x3072x768 tile 256x256x256 pipe=1 acc=0 all | 1 | 301,402 | 277839 | +8.5 | dma: 12 tiles/cluster, 1 clusters streaming, tile 4758 vs load 4512 |
| add_bias 256x3072 all | 1 | 72,225,418 | 72039662 | +0.3 | scalar: 13 blocks, 256 rows on the critical cluster (block 21 rows) |
| gelu 256x3072 all | 1 | 116,065,476 | 115188510 | +0.8 | scalar: 13 blocks, 256 rows on the critical cluster (block 21 rows) |
| gemm 256x768x3072 tile 256x256x256 pipe=1 acc=0 all | 1 | 204,809 | 199071 | +2.9 | dma: 3 tiles/cluster, 1 clusters streaming, tile 4758 vs load 4512 |

**{'noc_link_width': 512, 'redmule_ce': '64x64', 'mesh': '1x1'}**

| kernel | count | model cyc | sim cyc | err % | model note |
|---|---|---|---|---|---|
| layernorm 256x768 all | 2 | 59,743,210 | 56336592 | +6.0 | scalar: 4 blocks, 256 rows on the critical cluster (block 85 rows) |
| gemm 256x768x768 tile 256x256x256 pipe=1 acc=0 all | 4 | 74,957 | 69627 | +7.7 | dma: 3 tiles/cluster, 1 clusters streaming, tile 4694 vs load 4512 |
| add_bias 256x768 all | 5 | 18,056,969 | 18036324 | +0.1 | scalar: 4 blocks, 256 rows on the critical cluster (block 85 rows) |
| transpose 256x768 all | 1 | 3,261,100 | 1138733 | +186.4 | scalar: 48 blocks/cluster |
| gemm 256x256x64 tile 256x256x64 pipe=1 acc=0 c0 | 12 | 9,290 | 8180 | +13.6 | dma: 1 tiles/cluster, 1 clusters streaming, tile 1622 vs load 1417 |
| softmax 256x256 c0 | 12 | 21,829,043 | 20451151 | +6.7 | scalar: 1 blocks, 256 rows on the critical cluster (block 256 rows) |
| gemm 256x64x256 tile 256x64x256 pipe=1 acc=0 c0 | 12 | 13,790 | 12578 | +9.6 | dma: 1 tiles/cluster, 1 clusters streaming, tile 4694 vs load 2976 |
| add 256x768 all | 2 | 18,041,817 | 18025743 | +0.1 | scalar: 5 blocks, 256 rows on the critical cluster (block 56 rows) |
| gemm 256x3072x768 tile 256x256x256 pipe=1 acc=0 all | 1 | 299,098 | 275079 | +8.7 | dma: 12 tiles/cluster, 1 clusters streaming, tile 4694 vs load 4512 |
| add_bias 256x3072 all | 1 | 72,225,418 | 72039662 | +0.3 | scalar: 13 blocks, 256 rows on the critical cluster (block 21 rows) |
| gelu 256x3072 all | 1 | 116,065,476 | 115188510 | +0.8 | scalar: 13 blocks, 256 rows on the critical cluster (block 21 rows) |
| gemm 256x768x3072 tile 256x256x256 pipe=1 acc=0 all | 1 | 202,505 | 196155 | +3.2 | dma: 3 tiles/cluster, 1 clusters streaming, tile 4694 vs load 4512 |

**{'noc_link_width': 256, 'redmule_ce': '128x32', 'mesh': '1x1'}**

| kernel | count | model cyc | sim cyc | err % | model note |
|---|---|---|---|---|---|
| layernorm 256x768 all | 2 | 59,755,498 | 56346602 | +6.0 | scalar: 4 blocks, 256 rows on the critical cluster (block 85 rows) |
| gemm 256x768x768 tile 256x256x256 pipe=1 acc=0 all | 4 | 110,923 | 104070 | +6.6 | dma: 3 tiles/cluster, 1 clusters streaming, tile 4758 vs load 8608 |
| add_bias 256x768 all | 5 | 18,069,353 | 18047738 | +0.1 | scalar: 4 blocks, 256 rows on the critical cluster (block 85 rows) |
| transpose 256x768 all | 1 | 3,267,244 | 1144973 | +185.4 | scalar: 48 blocks/cluster |
| gemm 256x256x64 tile 256x256x64 pipe=1 acc=0 c0 | 12 | 10,826 | 9732 | +11.2 | dma: 1 tiles/cluster, 1 clusters streaming, tile 2134 vs load 2441 |
| softmax 256x256 c0 | 12 | 21,831,091 | 20453207 | +6.7 | scalar: 1 blocks, 256 rows on the critical cluster (block 256 rows) |
| gemm 256x64x256 tile 256x64x256 pipe=1 acc=0 c0 | 12 | 14,366 | 13154 | +9.2 | dma: 1 tiles/cluster, 1 clusters streaming, tile 2710 vs load 5536 |
| add 256x768 all | 2 | 18,060,249 | 18042413 | +0.1 | scalar: 5 blocks, 256 rows on the critical cluster (block 56 rows) |
| gemm 256x3072x768 tile 256x256x256 pipe=1 acc=0 all | 1 | 442,961 | 412851 | +7.3 | dma: 12 tiles/cluster, 1 clusters streaming, tile 4758 vs load 8608 |
| add_bias 256x3072 all | 1 | 72,275,818 | 72090638 | +0.3 | scalar: 13 blocks, 256 rows on the critical cluster (block 21 rows) |
| gelu 256x3072 all | 1 | 116,114,628 | 115237662 | +0.8 | scalar: 13 blocks, 256 rows on the critical cluster (block 21 rows) |
| gemm 256x768x3072 tile 256x256x256 pipe=1 acc=0 all | 1 | 344,158 | 328716 | +4.7 | dma: 3 tiles/cluster, 1 clusters streaming, tile 4758 vs load 8608 |

**{'noc_link_width': 256, 'redmule_ce': '64x64', 'mesh': '1x1'}**

| kernel | count | model cyc | sim cyc | err % | model note |
|---|---|---|---|---|---|
| layernorm 256x768 all | 2 | 59,755,498 | 56346602 | +6.0 | scalar: 4 blocks, 256 rows on the critical cluster (block 85 rows) |
| gemm 256x768x768 tile 256x256x256 pipe=1 acc=0 all | 4 | 110,731 | 103887 | +6.6 | dma: 3 tiles/cluster, 1 clusters streaming, tile 4694 vs load 8608 |
| add_bias 256x768 all | 5 | 18,069,353 | 18047738 | +0.1 | scalar: 4 blocks, 256 rows on the critical cluster (block 85 rows) |
| transpose 256x768 all | 1 | 3,267,244 | 1144973 | +185.4 | scalar: 48 blocks/cluster |
| gemm 256x256x64 tile 256x256x64 pipe=1 acc=0 c0 | 12 | 10,314 | 9220 | +11.9 | dma: 1 tiles/cluster, 1 clusters streaming, tile 1622 vs load 2441 |
| softmax 256x256 c0 | 12 | 21,831,091 | 20453207 | +6.7 | scalar: 1 blocks, 256 rows on the critical cluster (block 256 rows) |
| gemm 256x64x256 tile 256x64x256 pipe=1 acc=0 c0 | 12 | 16,350 | 15138 | +8.0 | dma: 1 tiles/cluster, 1 clusters streaming, tile 4694 vs load 5536 |
| add 256x768 all | 2 | 18,060,249 | 18042413 | +0.1 | scalar: 5 blocks, 256 rows on the critical cluster (block 56 rows) |
| gemm 256x3072x768 tile 256x256x256 pipe=1 acc=0 all | 1 | 442,193 | 412119 | +7.3 | dma: 12 tiles/cluster, 1 clusters streaming, tile 4694 vs load 8608 |
| add_bias 256x3072 all | 1 | 72,275,818 | 72090638 | +0.3 | scalar: 13 blocks, 256 rows on the critical cluster (block 21 rows) |
| gelu 256x3072 all | 1 | 116,114,628 | 115237662 | +0.8 | scalar: 13 blocks, 256 rows on the critical cluster (block 21 rows) |
| gemm 256x768x3072 tile 256x256x256 pipe=1 acc=0 all | 1 | 343,966 | 328290 | +4.8 | dma: 3 tiles/cluster, 1 clusters streaming, tile 4694 vs load 8608 |
