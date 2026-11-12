# NL2SQL-LoRA

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-50%20passed-brightgreen.svg)](tests/)

在**单张 RTX 4090（24GB）**上，用 4-bit QLoRA 把 Qwen2.5-7B 微调成 NL2SQL 模型，
并用 SQLite **真实执行结果**评测（不是文本相似度）。

> **Spider dev 执行准确率：66.15% → 76.11%（+9.96 点）**，精确匹配率 16.3% → 48.6%，
> 仅训练 0.92% 的参数，2 小时 24 分训完。

---

## 结果

| 指标 | 基座 Qwen2.5-7B | + QLoRA v1 | + QLoRA v2 |
|---|---|---|---|
| **执行准确率 (EA)** | 66.15% | 75.92% | **76.11%** |
| 精确匹配 (EM) | 16.34% | 48.55% | 48.55% |
| SQL 合法率 | 93.52% | 95.07% | 95.36% |
| — easy (476) | 78.36% | 87.61% | 87.61% |
| — medium (425) | 59.06% | 72.47% | 72.24% |
| — hard (133) | 45.11% | 45.11% | 47.37% |

**EM 只有 48.6% 而 EA 有 76.1%** —— 一半的 SQL 写法和标准答案不同，但查询结果正确。
这就是为什么必须用执行准确率评测：用文本相似度会低估 30 个百分点。

训练成本：854 / 1062 步，1h50m / 2h24m，显存峰值 10.9 / 18.9 GiB，40.37M 可训练参数。

## 快速开始

```bash
conda create -n nl2sql python=3.10 -y && conda activate nl2sql
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
python scripts/00_check_env.py          # 自检：CUDA / 显存 / bitsandbytes / 磁盘
```

数据（Spider 1.0 解压到 `data/spider/spider_data/`）：

```bash
python scripts/01_prepare_data.py --source spider --spider-dir data/spider/spider_data
# [train] kept=8647  [dev] kept=1034
# gold SQL 会真跑一遍，执行失败的自动丢弃
```

训练（约 2.5 小时）：

```bash
python -m nl2sql.train --config configs/qlora_7b.yaml \
  --train-file data/spider_proc/train.jsonl --dev-file data/spider_proc/dev.jsonl \
  --epochs 2 --output-dir outputs/qwen2.5-7b-spider-v2
```

推理 + 评测：

```bash
python -m nl2sql.infer --config configs/qlora_7b.yaml \
  --test-file data/spider_proc/dev.jsonl --batch-size 4 \
  --adapter outputs/qwen2.5-7b-spider-v2 --output outputs/pred.jsonl

python -m nl2sql.evaluate --pred outputs/pred.jsonl --out-dir outputs/report
```

没有 GPU 也能跑管线自检和数据准备：

```bash
python -m nl2sql.train --config configs/qlora_7b.yaml --dry-run   # 只验数据，不加载模型
python -m pytest -q                                                # 50 个用例，纯 CPU
```

## 工程结构

```
src/nl2sql/
├── schema.py       sqlite → DDL + 抽样值 + 外键，带 LRU 缓存
├── prompt.py       system/user 模板（训练与推理共用同一套 messages）
├── data.py         tokenize + 只对 SQL 算 loss 的 mask + collator
├── train.py        transformers.Trainer + PEFT，QLoRA 训练
├── infer.py        批量生成 + 执行结果自一致性投票
├── evaluate.py     EA / 合法率 / 分层 / 失败归因
├── merge.py        adapter → 完整模型
├── sql_utils.py    抽取、只读执行、结果集等价比较
└── synth.py        合成 demo 库（5 分钟冒烟测试）

scripts/
├── 00_check_env.py     环境自检
├── 01_prepare_data.py  Spider → jsonl（校验 gold SQL 可执行）
└── 03_compare.py       配对比较 + McNemar 精确检验
```

## 三个关键设计

**只对 SQL 部分算 loss。** prompt 的 token 全部 mask 成 `-100`。否则 7B 会把容量浪费在
复述 schema 上，收敛明显变慢。

**评测执行结果，不评测文本。** 比较前做归一化：无 `ORDER BY` 时忽略行序、
浮点容差 1e-6、`NULL` 与空串区分、大小写与空格归一。`gold` 带 `ORDER BY` 时才比较行序。

**SQL 执行有安全闸。** SQLite 以 `mode=ro` 只读打开，语句层拦截
`INSERT/UPDATE/DELETE/DROP/PRAGMA`，10 秒超时。基座模型确实生成过 1 条写操作，被拦下了。

## 两个被证伪的假设

好的实验不只报正结果。这个项目里两条看似合理的改进路径都被数据否掉了：

**1. 扩充数据库覆盖**（hard 样本 +66%，140 → 160 个库）
→ hard 只从 45.11% 到 47.37%（60 → 63/133），**假设证伪**。
hard 的瓶颈不是数据覆盖，是 7B 对复杂逻辑的推理上限。

**2. 执行结果自一致性**（采样 5 条按结果指纹投票）
→ hard 47.37% → 51.88%，修好 9 条、弄坏 3 条，**McNemar p = 0.146，不显著**。
5 倍推理成本换不显著收益，主动否决。

完整分析见 [REPORT.md](REPORT.md)。

## 失败分析

v2 的 247 条错误里：

- **199 条 `wrong_result`**：SQL 能跑但结果不对。JOIN 关系错、聚合维度错、
  AND/OR 优先级错。格式没问题，是语义错。
- **48 条 `invalid_sql`**：表名/列名臆造或语法错。比基座的 66 条减少 27%。

## 局限

- 只支持 SQLite 方言；MySQL/PostgreSQL 需要改 SQL 方言模板和评测连接器。
- 单表/两表查询接近可用（easy 87.6%），复杂嵌套查询（hard 47.4%）仍需 schema linking 等结构改进。
- 评测集是 Spider 的公开 dev 集，与训练集同一批数据库家族；在完全未见过的业务库上会低于这个数字。

## 环境

已验证：RTX 4090 24GB / Driver 580 / CUDA 13.0 / torch 2.14.1+cu130 /
transformers 4.57.6 / peft 0.21.2 / bitsandbytes 0.50.2 / Python 3.10。

## License

[Apache-2.0](LICENSE)。基座模型遵循 [Qwen 许可](https://github.com/QwenLM/Qwen2.5)。
