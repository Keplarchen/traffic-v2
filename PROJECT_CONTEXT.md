# Project Context — GEANT 流量预测 + 自适应带宽分配

> 给 Claude 看的项目上下文。新设备/新对话时让 Claude 先读这份，30 秒能接上之前的工作.

---

## 1. 项目定位

**任务**: 在 GEANT 骨干网 4 个月流量数据上, 同时预测 462 个 SD 对的未来流量, 并给出可直接用于带宽分配的**多分位数预测** (P50, P90, P95).

**核心目标**: P95 配额 = 覆盖 95% 实际流量, 工程语义为"5% SLA 违约率".

**对比方法**: Transformer (multivariate, 我们自己设计) vs PatchTST (channel-independent, 官方源码) vs 4 个传统 baseline.

---

## 2. 数据

- **来源**: `directed-geant-uhlig-15min-over-4months-ALL.tgz` (在 project/ 根目录, 不在 traffic-v2/ 里)
- **节点**: 22 个国家级节点
- **SD 对**: 22² − 22 自环 = **462 个有向流量对**
- **粒度**: 15 分钟
- **总样本**: **11460 时刻** (约 120 天, 2005-05-04 ~ 2005-08-31)
- **训练样本** (受 multi-resolution token 4-周 buffer 限制):
  - train: 6480 (Transformer) / 8481 (PatchTST, 只需 7-天 buffer)
  - val: 1131 ~ 1146
  - test: 1131 ~ 1146

### 数据 pipeline
```
.tgz
  │ core/parse_xml.py
  ▼
data/flows.npy        [11460, 462]  (Mbps)
data/timestamps.npy   [11460]       datetime64[m]
data/sd_pair_names.npy [462]        'src->tgt' 字符串
  │ core/clean_outliers.py
  ▼
data/flows_clean.npy  [11460, 462]  (clip > 10 Gbps + 线性插值,
                                    307 个异常 cell, 集中在每月 27 日,
                                    疑似 SNMP counter overflow)
  │ core/prep.py
  ▼
data/processed/all.pt  {
  flows_z:    [11460, 462]   log1p + z-score (仅 train fit)
  time_feats: [11460, 4]     sin/cos hour + sin/cos dow
  mean, std:  [462]          反变换用
  train_idx, val_idx, test_idx   80/10/10 时间序切分
}
```

---

## 3. 关键架构决策 (含理由)

### 3.1 Pinball loss, 不是 MSE / AsymmetricMSE
- **理由**: Koenker-Bassett 1978, 最小化 pinball 收敛到条件 τ-quantile
- **替代品的问题**: AsymmetricMSE penalty 系数对应不到具体 SLA, 梯度不稳
- 同时训三个分位数 τ ∈ {0.5, 0.9, 0.95}, .mean() 等权聚合

### 3.2 Multi-resolution context (Transformer 用)
- 107 token = 96 recent (24h) + 7 daily anchor + 4 weekly anchor
- 显式给模型提供日/周周期信号, 不用让它从头学
- 用 token-type embedding 区分 3 类
- **PatchTST 不用这个**, 走纯连续 7 天 (vanilla)

### 3.3 端到端残差 (Transformer 用)
- `out = current_z + delta`
- 强先验: 未训练时输出 = current_z, 等价 Naive Last
- 把 v1 (绝对值预测) 的 SLA 7.6%/Util 41% 改善到 v2 (delta) 的 4.9%/70%
- **PatchTST 不用**, 因为 RevIN 已经提供类似机制 (反归一化时加回 mean)

### 3.4 多分位数同时预测
- 输出 [B, 462, H, Q], 一次前向产 3 个分位数
- 不是用 Q=1 + 后处理乘安全系数 (那是统计学上不严谨的)

### 3.5 Single-horizon vs Multi-horizon (MIMO)
- 试验过 single h=16 vs MIMO h=(1,4,16)
- **结论**: MIMO 略好 (cross-task regularization), 不是用户最初担心的"MIMO 是包袱"
- 当前 PatchTST 用 single h=16 (可改回 MIMO)

---

## 4. 文件组织

```
traffic-v2/
├── core/                        # 跟模型无关的共享代码
│   ├── parse_xml.py             # tgz → flows.npy
│   ├── clean_outliers.py        # outlier interpolation
│   ├── prep.py                  # log1p + scaler + split
│   ├── dataset.py               # WindowDataset (multi-resolution, 给 Transformer 用)
│   ├── patchtst_dataset.py      # PatchTSTDataset (continuous, 给 PatchTST 用)
│   ├── losses.py                # pinball_loss (维度自适应)
│   ├── metrics.py               # compute_metrics + inverse_transform
│   └── baselines.py             # 4 个 baseline (multi-horizon 兼容)
│
├── models/
│   ├── transformer.py           # TrafficTransformer (我们设计的)
│   ├── patchtst.py              # PatchTST wrapper (薄薄一层)
│   └── patchtst_official/       # 官方源码 (PatchTST_backbone.py 等)
│
├── train_transformer.py
├── train_patchtst.py
│
├── eval/
│   ├── transformer/             # run_eval.py + plots.py + 输出
│   └── patchtst/                # 还没写
│
├── data/                        # parsing 输出
├── checkpoints/                 # 训练输出
└── PROJECT_CONTEXT.md           # 你正在读的这份
```

---

## 5. 实验进展 & 关键 findings

### Transformer (我们设计的) 实验链

| 版本 | 设计 | test SLA | test Util | 关键观察 |
|---|---|---|---|---|
| v1 absolute | 直接预测 P95 | 7.62% | 41.6% | 既欠覆盖又过分配 (架构信息瓶颈) |
| v2 delta | 加端到端残差 | 4.89% | 70.7% | **追平 Naive Last**, MAE_P50 从 42→9 Mbps |
| v3 MIMO | 多 horizon (1,4,16) | h=1: 5.5%, h=16: 7.2% | h=1: 73%, h=16: 48% | 长 horizon 校准变差 |
| Single h=16 | 单任务 4h 预测 | 5.79% | 36.7% | **比 v3 MIMO 还差** → MIMO 不是包袱 |

### Naive Last (residual P95) 是隐藏的强王
- 三个 horizon 全赢:
  - h=1: SLA 4.1%, Util 75%
  - h=4: SLA 4.1%, Util 65%
  - h=16: SLA 4.0%, Util 50%
- 自动 per-SD-pair 校准 (用 train 残差 P95 当 buffer)
- **GEANT 骨干流的天花板**: 平滑聚合流量 + 高自相关性 → naive 几乎最优

### 推论
- ML 在我们当前 multivariate 设计下被天花板压住
- **PatchTST 是当前最有希望突破的方向**: channel-independent + RevIN, 能把"视为独立 462 个序列"作为强归纳偏置
- 如果 PatchTST 也输 Naive Last → 是 GEANT 数据本质限制, 报告里诚实写

---

## 6. 当前进度 (写于 2026-05-09)

### 完成
- ✅ 数据 pipeline (parse, clean, prep, dataset)
- ✅ Transformer 多版本实验 (v1 → v2 → v3 → single h=16)
- ✅ 4 个 baseline + 完整 eval framework
- ✅ PatchTST 架构: 下载官方源码 + wrapper (`models/patchtst.py`)
- ✅ `core/patchtst_dataset.py`
- ✅ `train_patchtst.py`

### Pending
- ⏳ 跑 PatchTST 训练 (`python train_patchtst.py`)
- ⏳ 写 `eval/patchtst/run_eval.py` + `plots.py` (复制 transformer/, 改 import 即可)
- ⏳ 跑 PatchTST eval, 跟其他方法对比
- ⏳ 决定是否做 PatchTST 自监督预训练 (mode B)

### 未来可能的方向
- DLinear 对比 (channel-independent + 极简)
- LSTM 对比 (递归)
- TimesFM/Chronos 等 foundation model 微调

---

## 7. 实操注意事项

### 依赖
完整列表在 [`requirements.txt`](requirements.txt). 核心 4 个包:
- `torch>=2.6.0` (**装 GPU 版**, 见 requirements.txt 注释)
- `numpy>=2.0`
- `scikit-learn>=1.5`
- `matplotlib>=3.10`

### Python 环境
```powershell
conda create -n nyu_te python=3.11
conda activate nyu_te
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

之后跑命令前都要先 `conda activate nyu_te`. 或者用绝对路径:
```
D:/App/Anaconda/envs/nyu_te/python.exe path/to/script.py
```

**验证 GPU**:
```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 应该看到: True / NVIDIA GeForce RTX 4060 Laptop GPU (或同等 GPU)
```

### Batch size
- Transformer: 64 (default)
- PatchTST: **16** (channel-independent 让 effective batch = 16 × 462 = 7392, 显存边界)

### 训练时长
- Transformer: 2-3 sec/epoch, 总训练 1-3 分钟
- PatchTST: 预估 10-20 sec/epoch, 总训练 5-10 分钟

### 关键文件不能少
- `data/processed/all.pt` 必须存在 (跑 prep.py 生成)
- `checkpoints/transformer_best.pt` 是 single h=16 模型
- `checkpoints/patchtst_best.pt` 跑完 train_patchtst.py 才有

---

## 8. 给新对话 Claude 的指令

如果用户用新对话/新设备继续这个项目:
1. **先读这份 `PROJECT_CONTEXT.md`** 了解全貌
2. **不要重做已完成的事** (查 §6 进度表)
3. **架构决策不要回退** (查 §3 决策理由)
4. **数据是确定的** (查 §2), 不要重新讨论 SD 对数量等基础事实
5. 回答用户时, 引用本文件的具体节标号 (§3.1, §5 等) 让用户知道你看过

---

*Last updated: 2026-05-09. Update this file when major decisions / experiments happen.*
