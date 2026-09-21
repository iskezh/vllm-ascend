# BlasST 算子重构开销分析

> 基线：`blasst_pr_mod2_squash` @ `33be61a7f`（= PR #16610 Round 2 内容）
> 日期：2026-09-21 · 分支：`blasst-refactor-analysis`

## 0. TL;DR

- BlasST 总量 **~10.5k 行**，但 BlasST **真正特有的业务逻辑只有 ~1.1k 行**（kernel skip-block ~330 + tiling LPT/sparse ~250 + Python 集成 ~480）。其余 ~9k 行是"在 NPU 上落地一个自定义 AscendC attention 算子"的固有成本：模板库 5.6k、FlashDecode split 机器 ~730、aclnn/NNopbase/torch 五层转发样板 ~1k、host 校验/注册样板 ~450、测试 ~250。
- **安全瘦身空间 ~1k 行（~10%）**：L0 零风险删除 ~700 行 + L1 确认后删除 ~280 行，工作量 **2~3 人日**（含构建与验证）。
- **结构性重构**（压缩转发层、attn_infra 与 FIA 系共享、删 FD）理论上可再减 2~3k 行，但工作量 1~2 周且需全量回归，**不建议在 PR 评审期做**。
- 结论：代码量大头不是"写多了"，而是算子固有复杂度；重构收益有限，建议只做 L0+L1 瘦身，把评审叙事放在"BlasST 增量只有 ~1.1k 行核心业务逻辑"上。

## 1. 代码量全景

| 部分 | 行数 | 说明 |
|---|---|---|
| `op_kernel/attn_infra/*.hpp`（15 文件） | 5,650 | CATLASS 风格 GEMM/epilogue 模板库 |
| `op_kernel/`（entry/kernel.h/common/tiling_key） | 1,353 | kernel 主类 + 20 个 tiling key 分派 |
| `op_host/`（tiling/proto/def/aclnn） | 2,066 | tiling 1,225 + 校验/注册/aclnn 样板 |
| `blasst_attention_score_torch_adpt.h` | 283 | torch 侧 3 层包装 |
| `csrc/torch_binding*.cpp` | 148 | op 注册 + meta，纯样板 |
| **C++ 小计** | **~9,500** | |
| `vllm_ascend/attention/attention_v1.py` | ~590 | 集成（核心 ~480 + 可剥离统计 ~105） |
| `vllm_ascend/ascend_config.py` | 32 | `BlasstConfig` 5 个字段 |
| `tests/ut/`（两处） | ~280 | mock 化 UT |
| docs/build | ~75 | 配置文档 + CMake |
| **总计** | **~10,500** | 两个 commit（`617e84dbd` + `33be61a7f`） |

注：`vllm_ascend/_cann_ops_custom/vendors/` 下的同名文件**不是源码副本**，是 `csrc/build_aclnn.sh` 每次构建整体重生成的安装产物（`.gitignore` 忽略、git 只跟踪 `.gitkeep`）。当前主 checkout 的副本滞后 csrc 约 2 天提交（kernel 源 diff ~219 行：LPT 任务均衡、prefill causal 上界、dtype 换手等）。**重构不需要双向同步它，但任何 e2e 验证前必须重跑 `build_aclnn.sh`**，否则跑的是旧内核。

## 2. 成分分析：代码为什么这么多

### 2.1 attn_infra 模板库（5,650 行）——92% 在真实工作

逐文件沿模板实例化链确认：**没有任何一个 .hpp 整文件是死代码**。5 个 layout、3 个 CopyL0CToGm 特化、QK/PV 两个 BlockMmad、online softmax 双 operator() 全部在 20 个 tiling key 的实例化链上。死代码是文件内部的 FIA 残留：

| 死代码 | 行数 | 性质 |
|---|---|---|
| `coord.hpp` 未实例化成员（Argmin/算术/比较等） | ~190 | 模板成员从不实例化，删了无编译影响 |
| `epilogue_rescale_o.hpp` delStartRow/delEndRow 残留 | ~130 | FIA"删除行"特性，kernel 恒传 0/qSeqlen，运行期死分支 |
| 各文件死常量/死别名/未用成员 | ~150 | `BYTE_PER_BLK`、`L1BAlignHelper`、l1KDynamic 等 |
| **小计** | **~470（8%）** | |

### 2.2 20 个 tiling key 与 FD 机器

- 实例化组合：dtype{fp16,bf16} × paged{0,1} × mask{none,causal} × lse{none,out} = 16，另加 4 个 FlashDecode key（paged + lse none）。`INPUT_LAYOUT` 恒 TND、`ENABLE_UNIT_FLAG` 恒 false，是摆设参数。
- FlashDecode（split-KV）相关：tiling 侧 ~450 行（split 节点表/core clamp/decode 均分/prefill 切分）+ `combine_scale.hpp` 280 行（跨核 LSE reduce + O 加权合并）。其中 **prefill-FD 路径（`fillCoreInfoForPrefillFlashDecode`，173 行）当前不可达**——FD 门槛要求 `maxQSeqlen==1`，该函数只在 `maxQSeqlen>1` 时进入；注释里留有"prefill 走 FD 实测净亏损"的数据，疑似刻意保留备用。
- FD 由 `blasst_config.flash_decode`（默认 True）门控，是 decode 性能特性，不能整体删。

### 2.3 五层纯转发

```
python → torch_binding → npu_blasst_attention_score[_out]（torch_adpt.h）
       → fia_exec_common → EXEC_NPU_CMD_WS 宏 → dlsym aclnn → Nnopbase
```

torch_adpt.h 自身 3 层、aclnn cpp 346 行零业务逻辑（support-list 静态表 + 30 个 extern C + 宏）。这是 vllm-ascend 所有 custom op 的统一模式，BlasST 只是照做；且 `npu_blasst_attention_score_get_workspace`（72 行）手写展开了宏内逻辑，与宏重复。

### 2.4 host tiling（1,225 行）的成分

≈ **20% BlasST 新增**（LPT 代价均衡 124 行 + sparse ~50 + host-seq ~40 + FD 门槛 ~20）/ **37% FD 遗产** / **43% 通用校验注册样板**。明确死物：`GetKvNBlockTile` 死函数（8 行）；**12 个 tiling 字段 host 侧 set、kernel 从不读**（`mainLoopTaskNum/tailLoopTaskNum/numBlocks/maxQSeqlen/...`，~25 行）。

### 2.5 Python 集成（~590 行）

核心 ~480 行（能力矩阵+闸门 ~120 / eager forward ~60 / full-graph capture ~125 / update+replay 重绑 ~105 / helper ~65 / 分派 ~10），相互耦合——15 元组布局是 capture/update 双侧的隐式契约，不宜动。
**可剥离 ~105 行**：`_BlasstSkipStats` 统计体系（device 累加器 + 定期同步打日志），由 `collect_sparse_stats`（默认关）门控，可移到独立模块（如 `vllm_ascend/attention/blasst_stats.py`），让 attention_v1.py 回到 PR 评审友好的体量。

### 2.6 BlasST 真正的增量

kernel 侧 skip-block 逻辑 **~330 行**分布在 4 个文件：online_softmax 稀疏判定 ~140（lm-gm diff < λ → 整行早退）、rescale_o 稀疏半带加载/位图清零 ~120、gemm_pv 侧 DCCI+早退 ~25、kernel.h 调度/统计 ~40。**这才是评审者需要读懂的全部新算法**；其余都是移植的 FIA/attention 基础设施。

## 3. 瘦身方案分级与开销

| 级 | 内容 | 减行 | 改动量 | 风险 | 验证要求 |
|---|---|---|---|---|---|
| **L0** | 死代码删除：coord.hpp 成员 190 + rescale_o 死分支 130 + 死常量 150 + tiling 死字段/死函数 35 | **~505** | 0.5 人日 | 零（编译期可证） | 重编译 + UT + 单算子 replay |
| **L0+** | Python stats 剥离到独立模块 | ~105（attention_v1.py） | 0.5 人日 | 低（有 UT 兜底，stats 路径本身无 UT，需手测一次） | UT + eager 起服务各一次 |
| **L1** | 删不可达 prefill-FD 路径 173 + 清理 `ENABLE_UNIT_FLAG`/摆设参数 + `MASK_SPEC` 命名修正 | ~280 | 0.5~1 人日（含与作者确认是否保留 prefill-FD） | 低-中 | 同 L0 + graph 模式一次 |
| **L2a** | 转发层压缩：get_workspace 手写展开并入宏、torch_adpt 3 层→2 层 | ~100 | 1~2 人日 | 中（动所有调用路径） | 全量 UT + e2e eager/graph |
| **L2b** | 删 FD 整体（若确认不需要 flash_decode） | ~730 | 2~3 人日 | 高（性能回退风险，需 λ 对比数据支撑） | e2e 性能回归全套 |
| **L2c** | attn_infra 与 `sparse_attention_score` 的 FIA 系共享化 | 潜在 -3~4k | 1~2 周 | 很高（两套独立文件，需改 FIA 侧并回归 FIA） | 两个算子全量回归 |

**验证固定成本**（每次 C++ 改动都逃不掉，是"开销"的大头）：
1. `build_aclnn.sh` 全量构建——必须本地盘，NFS 时钟偏移会毁掉全量编译（见 memory）；构建后 vendor 安装包才与 csrc 一致；
2. E2E 服务器用 site-packages 副本，仓库 commit 后必须 `cp` 同步，否则跑旧代码；
3. 单算子 replay（2 号卡，FIA vs BlasST λ 对比）；
4. e2e eager + graph 各一轮（8 卡 TP8，64k TTFT/TPOT + ais_bench 16×65k）。

一轮完整 C++ 验证 ≈ 0.5~1 人日，与改动大小几乎无关——**这决定了"多次小步重构"比"一次改完"贵**，建议 L0/L1 合并成一次提交做一轮验证。

## 4. 建议

1. **PR 评审期：只做 L0 + L0+，一次提交一轮验证，减 ~600 行（其中 attention_v1.py 减 ~105 行），1.5~2 人日。** 风险零~低，直接回应"代码量太多"的评审意见。
2. L1 先问原作者 prefill-FD 是否刻意保留（注释里有实测数据，像是有意的），确认后可并入同一轮，再减 ~280 行。
3. L2 全部留到合入后。L2c（attn_infra 共享化）是唯一"大减行"项，但它本质是把两个算子绑进同一套模板库，需要独立 RFC 和 FIA 回归预算。
4. 评审沟通口径：BlasST 新增核心算法 **~1.1k 行**（kernel 330 + tiling 250 + python 480），其余是 AscendC attention 算子的固有基础设施（模板库/FD/转发样板），与 FIA 系算子同构。

## 5. 顺带发现（不在瘦身范围）

- `combine_scale.hpp:70`：`toUbTensor` 与 `broadCastOTensor` 用了同一 offset，疑似 bug（FD 合并路径），建议单独确认。
- `epilogue_rescale_o.hpp` `hmUbTensor` 分配后从未读取（~3 行，L0 可删）。
- `blasst_attention_score_tiling.h` 12 个字段 host set 而 kernel 不读，说明 tiling 结构体是从 FIA 整体拷贝后未裁剪——L0 删除时注意 `REGISTER_TILING_DATA_CLASS` 的字段声明要同步删，否则 tiling data 布局不一致会运行期炸（编译不出错），这是 L0 里唯一需要小心的点。
