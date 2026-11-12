export PYTHONPATH := src
PY ?= python
CONFIG ?= configs/qlora_7b.yaml
OUT ?= outputs/qwen2.5-7b-nl2sql

.PHONY: help env data smoke train baseline eval-base eval-lora merge test clean

help:
	@echo "make env      # 环境自检"
	@echo "make data     # 生成 demo 数据集（5 秒）"
	@echo "make smoke    # 30 步冒烟训练，验证显存/管线"
	@echo "make train    # 正式训练 ($(CONFIG))"
	@echo "make baseline # 基座模型推理（对照组）"
	@echo "make eval-base# 评测基座"
	@echo "make eval-lora# 评测 LoRA"
	@echo "make merge    # 合并 LoRA 权重"

env:
	$(PY) scripts/00_check_env.py

data:
	$(PY) scripts/01_prepare_data.py --source demo

smoke:
	$(PY) -m nl2sql.train --config $(CONFIG) --max-steps 30 --output-dir outputs/smoke

train:
	$(PY) -m nl2sql.train --config $(CONFIG) --output-dir $(OUT)

baseline:
	$(PY) -m nl2sql.infer --config $(CONFIG) --test-file data/demo/test.jsonl \
		--output outputs/pred_base.jsonl --limit 40

eval-base:
	$(PY) -m nl2sql.evaluate --pred outputs/pred_base.jsonl --out-dir outputs/report_base \
		--title "基座模型（未微调）"

eval-lora:
	$(PY) -m nl2sql.infer --config $(CONFIG) --test-file data/demo/test.jsonl \
		--adapter $(OUT) --output outputs/pred_lora.jsonl --limit 40
	$(PY) -m nl2sql.evaluate --pred outputs/pred_lora.jsonl --out-dir outputs/report_lora \
		--title "Qwen2.5-7B + QLoRA"

merge:
	$(PY) -m nl2sql.merge --base Qwen/Qwen2.5-7B-Instruct --adapter $(OUT) \
		--out outputs/merged

test:
	$(PY) -m pytest -q

clean:
	rm -rf outputs .pytest_cache
