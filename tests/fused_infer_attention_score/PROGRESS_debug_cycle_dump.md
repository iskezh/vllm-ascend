# FIA kernel 逐核 cycle 插桩 — 进度(2026-09-07 晚,第二 session 更新)

> 分支 `feature/custom-fia`,基线 commit `f2989fa88`。
> 本 session 已完成:重编全链路 + 采集 cycle dump。下一步:E2E 性能(64k,8 卡,4 组配置)。

## 状态:cycle 采集已完成 ✅;还原后 6b 崩溃已根因定位并修复 ✅

数据落盘:`tests/fused_infer_attention_score/cycle_dump_results_20260907.txt`(3 个 shape,解析脚本 `dump_cycle_stats.py`)。

## 基线 kernel stats 模式 cacheline 越界 bug(9-7 深夜定位)

**现象**:还原插桩(stats={2})后,套件内 fp16/blasst-lambda-3(6b)100% 崩
(EZ9999 aicore 507015 "MTE accesses an invalid GM address");6b 单独跑通过;
插桩版({4096})14/14 全绿。最小复现:`repro_6b.py 6a 6b`(golden→6a(stats)→6b)。

**根因**:kernel.h L465-483 stats 尾声,core0 在写完 `gSparseStatsOut[0/1]` 后执行
`DataCacheCleanAndInvalid<CACHELINE_OUT>` —— 回写整个 64B cacheline。{2} int32 输出
只有 8B,cacheline 里另外 56B 陈旧数据被写出到 GM,砸掉邻接内存:
- 邻居空闲 → 无感(部分布局通过)
- 邻居是活张量 → 数值损坏(任何 stats 启动后跟 6b 都 `finite=False`,与形状/跳过数无关,
  0/688 跳过也投毒 —— 排除跳过逻辑,锁定 stats 尾声)
- 邻居是下个 launch 的 seq-len/tiling → 地址飞 → "MTE invalid GM"

插桩版 {4096}=16KB,回写落在 tensor 内部,所以全绿 —— 插桩掩盖了基线 bug。
生产不受影响:attention_v1.py 4 处调用均 `sparse_stats_flag=False`(真跳过路径
不进 stats 尾声块)。但 stats 检测模式(阈值调参工作流)会触发。

**修复**:输出 {2}→{16} int32(64B 恰一 cacheline;NPU 分配器 512B 对齐保证不跨界)。
改动 4 处:torch_adpt.h L247/L308、proto.cpp SetDim(0,16)、torch_binding_meta.cpp。
消费方读 stats[0]/stats[1] 不变;生产侧全部 `_` 丢弃 stats。

**修复后验证(9-7 23:40)**:{16} 修复消除了数值投毒 —— 旧版任何 stats 启动后跟 6b 都
`finite=False`;修复版 stats+dense / stats+0跳过 → 6b 全部 finite=True。
bf16 关键路径 4/4 通过(dense/causal/paged-fd/真跳过);6b 单独跑 ×10 随机布局扰动全过。

**遗留(独立 bug,基线就存在)**:stats 启动**带非零跳过标志**(如 352/688)后紧跟
真跳过启动 → 后者 100% 崩(MTE invalid GM)。最小复现:`repro_6b.py 6a 6b`
(repro_6b.py 是套件忠实拷贝,可按用例裁剪)。触发要素:前一个 launch 是
stats 模式且实际算出了跳过标志;后一个 launch 走真跳过。0 跳过的 stats 启动
不触发;真跳过连续跑 ×10 不触发;单跑真跳过不触发。
生产不受影响:attention_v1.py 4 处 `sparse_stats_flag=False`,且每个进程 lambda
恒定,stats→skip 组合在 serving 中不可能出现。
疑似方向(未证):gSp 跨核标志/workspace 陈旧数据/跨 launch 硬件事件标志未复位,
真跳过路径读了上一 launch 的陈旧跳过标志导致地址计算飞掉。需要插桩 kernel 或
对照 aicore error 的 pc 值定位。

### 采集结论(定位 cube/vec 瓶颈)

| shape | cube 侧占比 | vec 侧占比 |
|---|---|---|
| sparse λ=-3 [640,1408] | pv 71%(165k/232k),qk 28% | softmax 43%,waitQk/waitPv 各 ~15%,spFlag 14% |
| dense causal 1×2048 | **pv 79%**(190k/239k),qk 19% | **softmax 49%**(234k/476k),spFlag 19%,waitPv 13% |
| dense causal 4×2048 | ⚠️ int32 溢出被 DbgClamp 钳位,无效 | 同左 |

→ cube 瓶颈在 **pv(matmul-value)**,vec 瓶颈在 **softmax**。多核负载均衡(每核 taskCycles 差 <10%)。
如需更大 shape 的数据,要把 dump 槽位改 int64 或分段累加(需改 kernel 重编)。

### E2E 性能验证(进行中,第 4 次启动 9-8 02:06 成功跑通)

**前 3 次失败与修复**(全部环境问题,非 kernel 问题):
1. 23:42 run#1:装的是旧 anchor(ba07e4a48)vllm 0.26.0+empty,repo HEAD 对
   e6bfe03ad → `VllmConfig has no attribute '_get_v1_model_runner_unsupported_features'`。
   修复:用 /mnt/share/vllm 的 e6bfe03ad 源码重编 empty wheel
   (`SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0 VLLM_TARGET_DEVICE=empty pip wheel`,
   需 `pip install setuptools_rust`),`--no-deps --force-reinstall` 装回。
2. 01:01 run#2:`/mnt/Qwen3-32B` 被清掉。修复:改用 `/home/weights/Qwen3-32B`
   (17/17 分片 62G 完整);`run_perf_pipeline.sh` MODEL_PATH 与 server 脚本 `--model` 都已改。
3. 01:22 run#3:server 正常,但压测数据集 `scripts/perf/built_in_dataset` 被清,
   6 个组合全 `ValueError: Dataset path ... not existed`,rc=0 但 0 CSV。
   修复:找到原始拷贝 `/home/z00603376/fia/perf/built_in_dataset`(8-24 15:57,
   与 8-25 成功跑一致)cp 回去。注意 65536.json 在 /home/weights tokenizer 下实际
   76920 token(+17%),81016<90000 不越界,4 组同文本对比不受影响。
   失败产物已归档 `blasst_res/perf/failed_runs_0908/`。

**run#4(orch_09080155)结果**:server 正常,warmup 通过,但 16×65536 组合时
server 崩溃(NPU 507011,MTE_ERR_0=0x3000025/0x28,pc dump:
`server/extra-info/data-dump/0/FusedInferAttentionScore_*.o`)。
**定性:崩的是 CANN 内置算子**(`/usr/local/Ascend/cann-9.1.0/opp/built-in/
.../ops_transformer/fused_infer_attention_score/`,sha256 匹配),不是我们的
custom 包(`vllm_ascend/_cann_ops_custom/vendors/custom_transformer`)。
根因:65536.json 在 /home/weights tokenizer 下实际 76920 token(+17%),
内置 FIA 在 KV>65536 区域 MTE 崩溃(38459 正常、76920 崩,边界在 64k)。
原数据集在旧 tokenizer 下恰好 65536 压线,换模型后越界。
**修复**:65536.json 修剪为**恰好 61440 token**(prompt+4096 输出 = KV 上限
恰 65536)。冒烟验证:baseline server 单请求 61440+4096 ignore_eos,HTTP 200,
567s,server 存活 —— 65536 KV 边界安全。16384/32768 文件不动(16096/38459 均正常)。
run#4 部分产物归档 `failed_runs_0908/`。

**run#5(orch_09080318)结果(06:39 结束)**:
- base:1 CSV —— 组合 1 后 server 又崩(03:46 decode stall → 03:55 507011,
  MTE 0x2803000025,故障 kernel 仍是**内置 FIA**)。内置路径两种触发:
  prefill 大 KV(run#4@76920)+ decode flaky(run#5@20k KV);16×16384 对比
  custom dense:总吞吐 1533→2935 tok/s(**+91%**),TPOT -48%,base 还有严重
  straggler(TP99 TPOT 0.159s / MAX_E2E 651s)。
- custom_dense:5/6 CSV。32×65536 失败:KV 仅用 10.8%(非容量),32 请求
  共享 479 block 前缀,decode 到 KV≈64.2k 整批 stall → HTTP 500;plog 报告
  故障 kernel 为 MatMulV2(decode graph bs32 内 MTE invalid GM,归因模糊)。
  **16×65536(61440 token)完整跑通** —— 64k 数据点有效。
- custom_-3:3/6 CSV(16k/32k/64k @bs16 全过,含 16×65536;32×16384 起 500)。
- custom_-1:0 CSV —— server 引擎初始化卡死(vector core timeout,06:41
  Worker_TP4 异常);待判:λ=-1 特有 vs 设备累积劣化(容器内无法 npu-smi
  reset;实测 chip0 小算子正常,AICore 100% 为残留计数)。

**run#6 补跑(orch_rerun_*,06:55 启动)**:custom_-1 全组 → custom_-3 全组
→ base 尝试,纪律同前(空载 600s)。-1 若再次卡启动则定性为 λ=-1 特有问题。
32×65536 为当前栈稳定性边界,对比表按 5 组合 + 16×64k 呈现。

## 最终结果汇总(9-8,全部 4 组完成)

数据来源:custom 组 = run#5 + run#6 补跑;dense 32×65536 两次均 stall(弃用);
base = run#5 的 16×16384 + run#4 归档(crash76920_base_09080206,16×16384/16×32768,
这两个长度的数据集文件未改过,协议有效)。base 共 4 次尝试得 2/1/0/0 个 CSV,
内置路径在此环境随机崩(prefill 大 KV 必崩 + decode 随机 stall),尽力而为。

**Total Token Throughput (tok/s)**:

| 组合 | base | custom dense | custom λ=-3 | λ=-3 vs dense | custom λ=-1 | λ=-1 vs dense |
|---|---|---|---|---|---|---|
| 16×16384 | 1533 | 2935 | 3404 | +16.0% | 3426 | **+16.7%** |
| 16×32768 | 3115 | 5758 | 5915 | +2.7% | 6011 | **+4.4%** |
| **16×65536** | 崩 | 7440 | 7640~7653 | **+2.9%** | 7833 | **+5.3%** |
| 32×16384 | — | 4741 | 4874 | +2.8% | 5016 | **+5.8%** |
| 32×32768 | — | 6685 | 7054 | +5.5% | 7236 | **+8.2%** |
| 32×65536 | — | 崩 | 8136 | — | 8418 | — |

- 16×16384:custom dense vs base **+91%**(base TPOT -48%,且 base 有严重 straggler:
  TP99 TPOT 0.159s、MAX_E2E 651s)。
- **64k 关键点(16×65536)**:dense 7440 → λ=-3 7640(+2.9%,两次复测 7653/7640,
  波动 0.2%)→ λ=-1 7833(+5.3%);TPOT 0.0340→0.0331→0.0323s。
- λ 越接近 0(跳过越激进)收益越大,排序 -1 > -3 > dense 在全部组合一致成立。
- 32×65536 上 λ=-3/-1 跑通(8136/8418)而 dense 崩 —— 当前栈的 bs32@64k 稳定性
  边界对 sparse 有利,非确定性边界。
- 复现性:λ=-3 组 16×65536 跨两轮 7653 vs 7640(0.2%),数据可信。

结论:custom FIA(含 {16} stats 修复)在 8 卡 TP8 E2E 全链路下,64k 长序列场景
较 dense 提速 2.9~5.3%,较 base(内置 FIA)在 16k 场景提速 91%,且稳定性显著优于
内置(base 4 次尝试均随机崩,custom 组 0 次崩溃)。

## host-list-only 方案 A' 第一步(9-8 晚,seq 通道切换 + dense A/B)

**改动**(不改 aclnn 签名/索引,分阶段降风险的第一步):
- tiling.h:`FAInferTilingData` 增加 `actualQSeq/actualKvSeq[256]` 内嵌数组
- tiling.cpp:host IntArray attrs 成为 seq 唯一来源(删 D2H fallback
  `CopySeqLengthsToHost`);tiling 落盘前写入数组;batch>256 硬检查
- kernel.h:seq 读取源 `params.actualQseqlen`(现传 nullptr)→
  `fATilingData->actualQSeq`,31 处 GetValue 调用点零改动
- torch_adpt.h:删 `upload_seq_lengths/seq_dev_view` 调用,设备 seq 输入传空
  optional(nullptr aclTensor,与 pse_shift 同款语义)
- attention_v1.py:删 update 窗口外 `preload_seq` 调用

**正确性验证**:
- validate_post_cleanup.py 14/14 全过(含 blasst-stats 352/688 golden 对齐)
- **6a+6b(stats→真跳过)×5 全过 —— 顺带修复了"遗留 bug"**(原:stats 启动带
  非零跳过后紧跟真跳过 100% MTE 崩)。机理吻合:设备侧 seq 常驻缓存正是疑似
  陈旧数据源,删除后病灶消失。
- 图模式冒烟:8k 上下文 needle 检索精确命中(chunked prefill eager 新路径 +
  paged decode capture/replay tiling 通道均正确)
- 宿主探针(/tmp/probe_tiling):tiling 序列化 6568B,q@2472/kv@4520 连续

**构建坑(重要)**:增量构建不重编 kernel——`csrc/build/binary/.../gen/*.done`
标记跳过 opc,`build/binary/.../src/` 下旧源码拷贝(还带插桩码)被沿用,`.o`
仍读 `params.actualQseqlen`(nullptr)→ invalid GM 崩。修复:删
`build/binary/ascend910_93/{src,gen,bin}` 下该算子产物强制重编;**验证手段:
kernel json 的 `opParaSize` 应等于新 tiling 尺寸+8(2480→6576)**。

**dense 组 A/B(09081827,同协议同纪律,6/6 CSV)**:

| 组合 | 基线 dense | hostlist dense | Δ吞吐 | TPOT |
|---|---|---|---|---|
| 16×16384 | 2935 | 3390 | **+15.5%** | 0.0267→0.0231 |
| 16×32768 | 5758 | 5794 | +0.6% | 0.0286→0.0284 |
| 16×65536 | 7440 | 7392 | -0.6% | 0.0340→0.0342 |
| 32×16384 | 4741 | 4748 | +0.1% | 0.0330→0.0330 |
| 32×32768 | 6685 | 6576 | -1.6% | 0.0492→0.0500 |
| 32×65536 | 崩(两次) | **跑通** | — | — |

- 16×16384 的 +15.5% 解读需谨慎:基线 dense 仅 run5 一个样本(该时段设备状态
  劣化:-1 挂起/dense stall 频发),部分增益可能是环境差异;也可能含真实的
  host 开销削减(preload 每步 ×64 层,内含 aclrtSynchronizeEvent)。
- 32×65536(hostlist 跑通 6/6):基线 dense 两次 stall 的组合一次通过。
- 64k 主点持平(-0.6%),无性能退化 —— host-list-only 的价值在架构简化
  (删 ~150 行设备侧管理 + 修复 stats→skip 遗留 bug + 与 base 接口契约对齐),
  而非性能。
- -3/-1 组 hostlist A/B 未跑(用户指示 dense 跑完即停)。

编排脚本 `kvcomp/scripts/pipeline/run_e2e_perf_all.sh`(后台运行,日志
`blasst_res/perf/orch_*/orchestrator.log`):每组 = 注入配置 → 杀旧 server →
卡空闲确认 → **空载 600s** → run_perf_pipeline.sh → 杀 server;顺序
base → custom_dense(-99) → custom_-3 → custom_-1。custom 组注入
`--additional-config '{"custom_fia_config":{"enabled":true,"full_graph":true,"sparse_lambda":X}}'`。
运行中的 kernel 构建 = 干净版 + {16} stats 修复(生产路径零差异)。
**E2E 期间不要重编 op/ext**(server 每组重启会加载 vendor,换包破坏对比)。
1. 8 卡空载 10 分钟后再启动(npu-smi 确认无进程)
2. 脚本:`/home/z00603376/blasst_model/kvcomp/scripts/pipeline/run_perf_pipeline.sh`
   (server 脚本 `launch_serving_nothink_full.sh` 已配 8 卡/TP8/max_model_len=90000,
   压测 run_test_full.sh 含 SEQ_LEN=65536)
3. 4 组:baseline / custom dense / custom λ=-3 / custom λ=-1。
   配置通过 `--additional-config '{"custom_fia_config":{"enabled":true,"sparse_lambda":X}}'`
   (见 vllm_ascend/ascend_config.py CustomFIAConfig;-99.0=dense,enabled 缺省 false=baseline)。
   注意 pipeline 的 `-t` 参数**只影响结果目录名**,不改变 server 配置,需要改 server 脚本或参数化。
   结果目录:`kvcomp/blasst_res/perf/{时间戳}_{阈值}_{模型}`。

### 还原插桩(执行 E2E 前)

`git checkout -- csrc/attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_kernel.h csrc/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_proto.cpp csrc/attention/fused_infer_attention_score/fused_infer_attention_score_torch_adpt.h csrc/torch_binding_meta.cpp vllm_ascend/attention/attention_v1.py`
然后重编 op 包 + 重装 vendor + 重建 ext(注意坑 4/5)。
**注意**:torch_binding_meta.cpp 里我补过一个 `}`(基线 commit 丢了 situ_mx_quant_meta 的收尾括号,见坑 2),checkout 后这个 bug 会回来,下次编译 ext 会再报错,需要重新补上(或该修复单独提交)。

## 本 session 踩坑记录(重要,复用)

1. **CANN 9.0.0→9.1.0 升级残留**:`csrc/build/CMakeCache.txt` 缓存旧路径导致 cmake 报
   `exe_graph ... cann-9.0.0 不存在`。修复:rm -rf csrc/build 全量重配(已完成,21:01 装包成功)。
2. **torch_binding_meta.cpp 基线就缺 `}`**:situ_mx_quant_meta 结尾右括号在 FIA 段插入时被挤掉,
   因 .so 自 8-31 未重编而漏网。症状:function-definition not allowed + expected '}' at end of input。
   已补(L2043 前);checkout 还原插桩时会回来,记得再补。
3. **op 构建与 ext 构建共用 csrc/build,不能并行**:CANN opc 的 kernel_meta.lock 按 pid 校验,
   并行必报 "Another process is using this dir"。串行执行。
4. **ext 的 kernels_preprocess merge 不可重入**:`merge_obj_text.sh` 用
   `ld.lld -m aicorelinux -Ttext=0 obj -o obj` 原地 REL→EXEC;custom target 每次构建都重跑,
   第二次对 EXEC 输入必报 "unknown file type"。修复:每次 ext 增量重建前先
   `rm -rf build/temp.linux-aarch64-cpython-311/vllm_ascend_kernels_preprocess-prefix`。
5. **editable 安装丢失**:pip show 不见了(import 全路径失败),用
   `pip install --trusted-host mirrors.huaweicloud.com --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi -v -e .`
   重装成功(22:01,该命令内部会跑 build_aclnn.sh + ext,全链路验证通过)。
6. **vendor 目录改名**:新包装到 `vendors/custom_transformer`(旧名 vllm-ascend)。
   - 生产代码 utils.py 的 `_CUSTOM_OP_VENDOR_DIR="custom_transformer"` 正确。
   - 测试脚本硬编码旧名的:validate_post_cleanup.py 已改;perf_driver/memcheck_driver/
     bench_host_vs_device 等还是旧名,用前要改。
   - **不要用软链兼容**:CANN 运行时对 symlink 路径报 EZ1009(op package not installed)。
7. `.so` 已含 FIA binding:9-7 22:01 重编,1.0MB(vllm_ascend_C.cpython-311),import 正常。
8. 精度回归 14/14 通过(validate_post_cleanup.py,含 blasst-stats 352/688 与 golden 对齐)。

## 原始插桩设计(保留参考)

dump 走 `sparse_stats` 输出 tensor(int32 {4096}),`sparse_stats_flag=true` 门控,非 FD 路径,
性能路径零改动。布局(kernel 头文件 ~L446 注释):
- `stats[0]=sparse_block_sum, stats[1]=total_block_sum`(cube core0 聚合)
- cube core c:`stats[64+c*32 .. +6]` = taskCycles, loadQ, qk, pv, stacks, tasks
- vec core c sub b:`stats[2112+c*64+b*32 .. +8]` = taskCycles, waitQk, softmax, spFlag,
  waitPv, rescale, stacks, tasks
- 每核 kernel 结束写一次累计值,DbgClamp 钳 int32;coreNum≤30 时最大偏移 4071 < 4096
- 涉及 5 个文件:kernel.h / proto.cpp({2}→{4096}) / torch_adpt.h / torch_binding_meta.cpp /
  attention_v1.py

## 环境备忘

- SoC ascend910_93,双路编译 `__DAV_C220_CUBE__`/`__DAV_C220_VEC__`。
- `AscendC::GetSystemCycle()` 可用。
- `csrc/build/binary/ascend910_93/src/` 是构建期源码拷贝,不要在那里改。
- 测试脚本自带 `ASCEND_CUSTOM_OPP_PATH` setdefault(validate 已指向 custom_transformer)。
