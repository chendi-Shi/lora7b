# Qwen2.5-7B NL2SQL QLoRA 微调报告

> 硬件：单张 RTX 4090 24GB ｜ 数据：Spider 1.0 ｜ 评测：Spider dev 1034 条，SQLite 真实执行

## 1. 结论

在单张 4090 上对 Qwen2.5-7B 做 4-bit QLoRA 微调（仅训练 0.92% 参数），
Spider dev **执行准确率从 66.15% 提升到 76.11%（+9.96 点）**，
精确匹配率从 16.34% 提升到 48.55%（+32.2 点）。
额外验证了两条改进路径（扩充数据库覆盖、自一致性解码），均未取得统计显著收益，
结论是 7B 模型在 Spider hard 档的能力上限约 47~52%。

| 指标 | 基座 Qwen2.5-7B | QLoRA v1 | QLoRA v2 |
|---|---|---|---|
| **执行准确率 (EA)** | 66.15% | 75.92% | **76.11%** |
| 精确匹配 (EM) | 16.34% | 48.55% | 48.55% |
| SQL 合法率 | 93.52% | 95.07% | 95.36% |
| — easy (476) | 78.36% | 87.61% | 87.61% |
| — medium (425) | 59.06% | 72.47% | 72.24% |
| — hard (133) | 45.11% | 45.11% | 47.37% |
| 结果错误 (wrong_result) | 283 | 198 | 199 |
| 非法 SQL (invalid_sql) | 66 | 51 | 48 |

## 2. 方法

### 2.1 训练配置

| 项 | 值 |
|---|---|
| 基座 | Qwen2.5-7B-Instruct（本地缓存，离线加载） |
| 量化 | 4-bit NF4，double quant，计算精度 bf16 |
| LoRA | r=16, alpha=32, dropout=0.05，作用于 q/k/v/o + gate/up/down |
| 可训练参数 | 40.37M / 4.39B（0.92%，4bit 打包后计数） |
| batch | 1 × grad_accum 16 |
| 序列长度 | 2048（丢弃 161 条超长样本，2.3%） |
| 学习率 | 2e-4，cosine，warmup 10% |
| epoch | 2 |
| 显存峰值 | v1 10.94 GiB / v2 18.87 GiB |
| 训练时长 | v1 1h50m（854 步）/ v2 2h24m（1062 步） |
| 框架 | transformers 4.57.6 + peft 0.21.2 + bitsandbytes 0.50.2，torch 2.14.1+cu130 |

### 2.2 三个关键实现细节

**只对 SQL 部分计算 loss。** prompt（system 指令 + schema + 问题）的 token 全部 mask 成 -100，
labels 只保留 assistant 回复。否则 7B 会把容量浪费在复述 schema 上，收敛明显变慢。

**评测用执行结果而非文本相似度。** 比较前对结果集做归一化：无 ORDER BY 时忽略行序、
浮点容差 1e-6、NULL 与空串区分、大小写与首尾空格归一。
这解释了 EM 只有 48.55% 而 EA 有 76.11% —— 一半的 SQL 写法不同但结果正确。

**SQL 执行有安全闸。** SQLite 以 `mode=ro` 只读打开，语句层面拦截
INSERT/UPDATE/DELETE/DROP/PRAGMA 等，并设置 10 秒超时（progress_handler 机制）。
基座模型确实生成过 1 条写操作，被拦下了。

### 2.3 数据

| | v1 | v2 |
|---|---|---|
| 训练样本 | 6988（train_spider.json） | 8647（+ train_others.json 1659 条） |
| 覆盖数据库 | 140 | 160 |
| hard 训练样本 | 1142 | 1894（+66%） |

数据转换时会用 sqlite 真实执行每条 gold SQL，执行失败的直接丢弃（v1 丢 12 条，0.17%）。

## 3. 实验一：扩充数据库覆盖（负结果）

v1 的结果里，easy +9.25、medium +13.41，但 **hard 一点没动（45.11% → 45.11%）**。

假设：hard 查询多涉及未见过的复杂库结构，140 个库不够，属于**数据覆盖问题**。

做法：合并 Spider 官方的 `train_others.json`（另外 20 个库、1659 条），
hard 训练样本增加 66%，其他超参完全不变（控制变量）。

结果：**假设被证伪**。hard 从 45.11% 到 47.37%（60 → 63 条，噪声范围内），
总体 +0.19 点。说明 hard 的瓶颈不是库的覆盖度，而是模型对复杂逻辑
（嵌套子查询、多表 JOIN、集合运算）的推理能力上限。

## 4. 实验二：执行结果自一致性（边际收益，未采用）

只用 v2 的 133 条 hard 样本测试：每条采样 5 次（temperature 0.7, top_p 0.95），
按**执行结果指纹**投票（不是按文本投票，等价 SQL 会自动合并成一张票）。

| | v2 贪心 | v2 + 自一致性 (k=5) |
|---|---|---|
| hard EA | 47.37% | **51.88%** |
| 正确数 | 63/133 | 69/133 |

配对比较（McNemar 精确检验，脚本 `scripts/03_compare.py`）：

| | v2 正确 | v2 错误 |
|---|---|---|
| **SC 正确** | 60 | 9（修好） |
| **SC 错误** | 3（弄坏） | 61 |

净增 6 条（修好 9、弄坏 3），**p = 0.146，不显著**。方向上自一致性确实在修
JOIN 别名错位、聚合维度错这类"手滑"，但样本量（12 条不一致样本）不足以证明收益，
而推理成本是 5 倍。**结论：不纳入最终方案，记录为已排除的路径。**

## 5. 失败分析

v2 的 1034 条中有 247 条错误：

- **199 条 wrong_result**：SQL 能执行、结果不对。主要是 JOIN 关系错、聚合维度错、
  多条件 AND/OR 优先级错。格式没问题，是语义错。
- **48 条 invalid_sql**：表名/列名臆造，或语法错误。相比基座的 66 条已减少 27%。

按难度看，easy 87.61% 已接近实用，medium 72.24% 可用，hard 47.37% 是主要短板。
自一致性修好的 9 条中，典型例子是 JOIN 时别名张冠李戴
（`T1.Model JOIN T2` 写成 `T2.Model JOIN T1`）——这类错误对一次贪心解码是致命的，
多次采样才有可能蒙对。

## 6. 复现

```bash
# 环境
conda create -n nl2sql python=3.10 -y && conda activate nl2sql
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .

# 数据（Spider 解压到 data/spider/spider_data/）
python scripts/01_prepare_data.py --source spider --spider-dir data/spider/spider_data

# 训练
python -m nl2sql.train --config configs/qlora_7b.yaml \
  --train-file data/spider_proc/train.jsonl --dev-file data/spider_proc/dev.jsonl \
  --epochs 2 --output-dir outputs/qwen2.5-7b-spider-v2

# 评测
python -m nl2sql.infer --config configs/qlora_7b.yaml \
  --test-file data/spider_proc/dev.jsonl --batch-size 4 \
  --adapter outputs/qwen2.5-7b-spider-v2 --output outputs/pred.jsonl
python -m nl2sql.evaluate --pred outputs/pred.jsonl --out-dir outputs/report

# 配对比较（两份预测文件之间）
python scripts/03_compare.py --a outputs/pred_a.jsonl --b outputs/pred_b.jsonl

# 合并权重
python -m nl2sql.merge --base Qwen/Qwen2.5-7B-Instruct \
  --adapter outputs/qwen2.5-7b-spider-v2 --out outputs/merged-v2
```

## 7. 下一步（按性价比排序）

1. **Schema linking**：先用向量检索筛出相关表/列，把长 schema 压短。
   hard 的错误里有一批是"选错了表"，这类问题靠采样投票救不回来，只能靠检索。
2. **换更大基座**：14B 的 QLoRA 在 24G 上放不下，需要 A100 40G 或双卡。
3. **约束解码**：接 SQLite grammar（outlines / xgrammar），把合法率从 95% 推到 ~100%，
   但对 wrong_result 无能为力，预计总体 +1~2 点。
4. **更长上下文**：`max_len` 3072 放回被截断的 161 条长 schema，预计对 hard 有少量帮助。
