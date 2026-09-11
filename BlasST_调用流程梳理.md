# custom FIA（BlasST）调用流程梳理 —— eager 与图模式

- **基准**：HEAD `c3297b919` + 工作区 W2 未提交改动（host-list-only 改造）
- **标注**：〔W2〕= 仅存在于工作区改造后的行为；〔⚠A*〕= 该步骤存在检视报告确认的缺陷

---

## 1. 分发总览

```
Attention 前向（每层每步）
    │
    ▼
forward_impl (attention_v1.py ~:2057)
    │
    ├─① DecodeOnly ∧ sliding_window=None ∧ using_paged_attention()
    │      → forward_paged_attention（baseline，优先级最高）
    │
    ├─② _can_use_custom_fia (:1438)  ──────────── custom FIA 接管
    │      条件（全部满足）：
    │        · custom_fia_config.enabled == True（默认 False）
    │        · capture 中：full_graph==True ∧ attn_state==DecodeOnly
    │          〔⚠A3 缺 draft 门控 / ⚠A4 缺 layer-aware 门控〕
    │        · sinks is None ∧ sliding_window is None
    │        · attn_state ∈ {DecodeOnly, PrefillNoCache, ChunkedPrefill}
    │          〔⚠A1 缺 causal 门控 → 非 causal 走此路径会静默算错〕
    │          〔⚠A2 缺 VLLM_BATCH_INVARIANT 门控 → 硬崩溃〕
    │          〔⚠A7 混合批未分相 → 数值随 batch 组成漂移〕
    │      │
    │      ├─ capture 中 → full_graph_custom_fia（→ §3）
    │      └─ 其余      → forward_custom_fused_infer_attention（→ §2）
    │
    └─③ → forward_fused_infer_attention（baseline torch_npu FIA）
```

注意 ② 的实际生效范围：由于 ① 优先，常规 decode 配置（using_paged_attention=True）下 enabled=True 也走不到 custom——decode 真正进 custom 的场景是 paged attention 关闭/不适用的配置，以及 PrefillNoCache / ChunkedPrefill（①只拦 DecodeOnly）。

---

## 2. Eager 路径（`forward_custom_fused_infer_attention`，:1471）

### 2.1 Python 侧准备

```
forward_custom_fused_infer_attention(query, key, value, attn_metadata, output)
    │
    ├─ _get_fia_params(key, value, attn_metadata)  (:1491，⚠D6 不传 kv_cache)
    │    按 attn_state 取：key/value 视图、block_size、block_table、actual_seq_lengths_kv
    │      · PrefillNoCache: block_table=None, kv_len=actual_seq_lengths_q（非 ED）
    │      · DecodeOnly/ChunkedPrefill: key_cache/value_cache 3D view + block_tables
    │    〔self.key_cache is None 时直接 RuntimeError；baseline 可经 kv_cache 懒初始化〕
    │
    ├─ num_tokens = actual_seq_lengths_q[-1]（累积和末项=总 token 数）
    ├─ query = query[:num_tokens]；PrefillNoCache 时 key/value 同步截断
    ├─ seq 列表：aq 直接用 list；akv 若是 Tensor 转 tolist()
    │    契约：aq = 前缀和；akv = 前缀和（非 paged）/ 每 batch 原始长度（paged）
    │
    └─ torch.ops._C_ascend.npu_fused_infer_attention_score(
           query, key, value, None(pse), attn_mask, aq, akv, block_table,
           num_heads, scale, SWA_INT_MAX, 2147483647, "TND", num_kv_heads,
           sparse_mode=3, inner_precise=0, block_size, antiquant_mode=0,
           sparse_lambda=<config>, lse_flag=False, stats_flag=False,
           host_seq_tiling=<config>, flash_decode=<config>)     ← 24 个参数
```

### 2.2 torch 适配层（`fused_infer_attention_score_torch_adpt.h`）

```
npu_fused_infer_attention_score(...)
    ├─ 分配输出：attention_out=query 同形、softmax_lse=[T,H,1] fp32、sparse_stats=[16] int32
    └─ fia_exec_common(...)
         ├─〔W2〕seq 处理：host-list-only —— 不再建 device seq 张量；
         │        device seq 输入以 absent optional（nullptr aclTensor）传入
         │        （DevListCache/upload_seq_lengths/seq_dev_view 成为死代码，待删）
         └─ workspace 未提供 → EXEC_NPU_CMD：
              ① aclnnVllmFusedInferAttentionScoreGetWorkspaceSize(...)
                 · aclnn wrapper 把 17 个 attr 逐一注册进 executor
                   （索引 12/13 = aq/akv host IntArray；15/16 = host_seq_tiling/flash_decode）
                 · op host 跑 tiling（见 2.3）→ 返回 workspace 大小
              ② aclnnVllmFusedInferAttentionScore(workspace, ..., stream) 真正下发
```

### 2.3 op host：tiling（`fused_infer_attention_score_tiling.cpp`，W2 后）

```
TilingVllmFusedInferAttentionScore(context)
    ├─ 取核数 coreNum = GetCoreNumAic()
    ├─〔W2〕seq 只认 host attrs：
    │    · TryGetSeqLengthsFromAttr(12/13) 必须成功，否则 OPS_LOG_E 报错（host-list-only）
    │    · batch > FIA_MAX_HOST_SEQ_LIST(256) → OPS_LOG_E（tiling 内置数组上限，显式报错）
    │    · device seq 输入 tensor 已改 optional（presence 双双一致即可，不参与计算）
    │    ·〔W2 前〕host attrs 缺失时可回退 CopySeqLengthsToHost（D2H）——已删除
    │    · host_seq_tiling attr（15）控制上述回退开关 →〔W2 后〕该 attr 事实上失去作用，待清理
    ├─ ConvertContextToFAInferContext → FAInferContext
    │    · sparse_mode==3 → MaskType::MASK_SPEC（恒定映射，无视 mask 是否为 None → ⚠A2 根因之一）
    │    · FD 门控（:767-771）：paged ∧ q全=1 ∧ blockSize=128 ∧ !lseFlag ∧ innerPrecise=0
    │      ∧ numTasks*5≤coreNum*4 ∧ numTasks≤26 ∧ minKv≥fdMinKv ∧ flash_decode attr
    │      〔⚠A8：核数未钳位到 MAX_CORE_NUM_FD=26，≥27 核 SoC 越界〕
    ├─ DoTiling：算 20 个 tiling key 之一（base=5e18 + paged/mask/lse/dtype/FD 组合）
    │    ·〔W2〕把 aq/akv 逐元素拷进 TilingData 内置数组 actualQSeq/actualKvSeq（:837）
    ├─ SaveToBuffer → 框架把 TilingData 上传 device（task-update 安全的载体）
    └─ workspace 大小 = FD split 区 + base 区推导
```

### 2.4 kernel（`fused_infer_attention_score_kernel.h`）

```
按 tiling key 选编译实例（双路编译 C220_CUBE/C220_VEC）
    ├─〔W2〕seq 读取源从 params.actualQseqlen（device GM 输入）
    │        改为 fATilingData->actualQSeq（框架托管的 tiling buffer）(kernel.h:126-130)
    ├─ SplitFuse 主循环：loadQ → QK(mm) → softmax(在线稀疏判定, sparse_lambda)
    │   → P(mm) → rescaleO，稀疏跳过受 sp_flag 门控（stats 模式 detection-only）
    ├─ FD 路径：split-KV + 跨核 LSE 合并（26 核数组布局）
    └─ stats 模式：stats[0]/[1] = skip/total 块计数，整 cacheline 回写
        （输出张量须 ≥16 int32 = 64B —— proto 已固定 {16}）
```

---

## 3. 图模式（ACL Graph + task-update 路径）

**启用前提**：`custom_fia_config.full_graph=True` + DecodeOnly bucket capture。其余 capture 场景回退 baseline（图内无真跳但正确）。

### 3.1 Capture（首次跑某 bucket，`full_graph_custom_fia`，:1545）

```
_get_fia_params → num_tokens（=bucket 大小）
    │
    ├─ 常驻输出三件套（capture/update/replay 复用同地址）：
    │     attention_out、softmax_lse、sparse_stats={2} 〔D2a：与 adapter/proto 的 {16} 不一致〕
    │
    ├─ 常驻 workspace（按 num_tokens bucket 缓存，只在 None 时分配）：
    │     ws_size   = get_workspace(真实 aq/akv 形状)          ← 只经 attr 推导，无 H2D，capture 期合法
    │     ws_size_fd = get_workspace(aq=1..b 累积, akv=max_model_len×b)   ← FD 上界
    │     取 max 常驻〔⚠A6：与 baseline 共用 workspaces 桶但无 cache_graph_workspace
    │        跨算子取大 → 混合层模型可能拿欠尺寸 buffer〕
    │
    ├─ 注册到 graph_params：events.append(ExternalEvent)
    │     attn_params.append(("custom_fia", <15 元组：weak_ref 张量×8 + 标量×7>)
    │     〔⚠D3：元组靠位置对齐，append(:1632) ↔ update(:841) 两处手工同步〕
    │
    └─ torch.npu.graph_task_group_begin(stream)
       npu_fused_infer_attention_score_out(query, key, value,
           attention_out, lse, stats, workspace, ..., 24 参同 eager)
       torch.npu.graph_task_group_end(stream) → handle 存 bucket
```

`_out` 变体与 eager 入口的差异：输出与 workspace 由调用方提供（地址恒定）；其余链路（aclnn→tiling→kernel）完全一致。

### 3.2 Replay（每个推理步）

```
ACLGraphRunner.replay (acl_graph.py:295)
    └─ update_graph_params(attn_metadata, ...)  (attention_v1.py:519)
         遍历每个 bucket 的 attn_keys → 对每条 "custom_fia" 参数（:835）：
           ├─ 取本步最新 metadata：
           │     seq_lens = attn_metadata[key].seq_lens_list
           │     aq        = attn_metadata[key].actual_seq_lengths_q
           │     block_tables = attn_metadata[key].block_tables
           │     〔⚠A4：裸 key 取值，无 baseline 的 layer_name 感知解析 → gemma4 类
           │       混合层模型跨层错配；⚠A3：未区分 draft/target 图池〕
           │     〔W2：preload_seq 窗口外刷 pinned 的步骤已删除〕
           │
           ├─ torch.npu.graph_task_update_begin(update_stream, handle)
           │     npu_fused_infer_attention_score_out(<新 aq/akv/block_tables + 捕获期张量>)
           │       · 窗口内是纯 aclnn launch：宿主列表经 attrs 进 tiling，
           │         TilingData 由框架随 task-update 重新托管上传 →〔W2〕无需任何
           │         窗口外 H2D 预热（旧方案的 pinned 缓存/507009 报错约束不复存在）
           ├─ torch.npu.graph_task_update_end(update_stream)
           └─ event.record(update_stream)
         随后 torchair 按 patched 参数重放整图 → kernel 从 tiling buffer 读最新 seq
```

### 3.3 图内禁用项

capture/update 期 `softmax_lse_flag=False ∧ sparse_stats_flag=False`（host 读 device 在 capture/replay 下非法）——图模式永远不产出 lse/stats。

---

## 4. W2 改造前后对比（seq 长度传递路线）

| 环节 | W2 前（c3297b919 提交版） | W2 后（当前工作区） |
|---|---|---|
| eager seq 上送 | adapter 内 DevListCache：pinned→dev copy_，shadow 去重，每变更 `aclrtSynchronizeEvent`〔⚠A5 容量 512 封顶、⚠A9 每步同步〕 | host IntArray attrs 直达 tiling，零 device 拷贝 |
| 图模式 seq 新鲜度 | 窗口外 `preload_seq` 刷 pinned（cache.dev 地址恒定约束） | attrs→TilingData，框架托管，窗口内直接 patch |
| op 的 device seq 输入 | 必填 int64 tensor | absent optional（保留接口位） |
| kernel seq 读取 | GM 输入张量 `params.actualQseqlen` | tiling buffer `fATilingData->actualQSeq` |
| batch 上界 | DevListCache TORCH_CHECK（512 起，不可增长） | tiling 内置数组 256，超限 OPS_LOG_E 显式报错 |
| `host_seq_tiling` attr | 控制 attrs/D2H 回退 | 回退已删 → attr 失去语义，待清理 |
| `preload_seq` op | 图模式必需 | 无调用方 → schema/impl 待删 |

**结论**：W2 落地后，⚠A5/⚠A9 自然消失（死代码删除即可），但引入新边界 **batch ≤ 256**（decode bucket 上限 96 + 余量，eager 混合大批需确认不超）；⚠A1/A2/A3/A4/A6/A7/A8 与 seq 传递无关，仍在。

---

## 5. 三层调用链速查

```
eager:
  attention_v1.forward_impl → forward_custom_fused_infer_attention (:1471)
    → torch.ops._C_ascend.npu_fused_infer_attention_score          [schema: torch_binding.cpp]
    → vllm_ascend::npu_fused_infer_attention_score                 [torch_adpt.h，分配输出]
    → fia_exec_common → EXEC_NPU_CMD
    → aclnnVllmFusedInferAttentionScore{GetWorkspaceSize,}         [op_api wrapper，17 attrs]
    → TilingVllmFusedInferAttentionScore                           [op_host，host-list→TilingData]
    → SplitFuse kernel（tiling-key 分发，seq 读 tiling buffer）

graph capture:
  forward_impl → forward_custom_fused_infer_attention
    → full_graph_custom_fia (:1545)：常驻输出+workspace(bucket) → 注册 attn_params
    → graph_task_group_begin → npu_fused_infer_attention_score_out → graph_task_group_end

graph replay（每步）:
  acl_graph.py:295 → update_graph_params (attention_v1.py:519)
    → "custom_fia" 分支 (:835)：取新 metadata
    → graph_task_update_begin → _out（attrs 随 tiling 重托管）→ end → event.record
    → ACL Graph 重放
```
