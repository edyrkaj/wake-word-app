# Python pipeline for wake-word training / export (see model_gen/).
# Override the interpreter: make train PY=.venv/bin/python
PY ?= python3
MODEL_GEN := model_gen
REQ := $(MODEL_GEN)/requirements.txt

.DEFAULT_GOAL := help

.PHONY: help py-install record train train-weights export-scaffold quantize-shrink clean-model-gen

help:
	@echo "Python / model_gen targets"
	@echo "  make py-install       pip install -r model_gen/requirements.txt"
	@echo "  make record           record_dataset.py  (pass RECORD_ARGS='...')"
	@echo "  make train            train_model.py then export quantized ONNX to public/"
	@echo "  make train-weights    train only; skip ONNX (--no-export)"
	@echo "  make export-scaffold  export_model.py (untrained arch + quantize to public/)"
	@echo "  make quantize-shrink  shkrink-mode.py (needs wake_word.onnx in model_gen/)"
	@echo "  make clean-model-gen  remove common artifacts under model_gen/"
	@echo ""
	@echo "Examples:"
	@echo "  make record RECORD_ARGS='--label my_wake_word --count 5'"
	@echo "  make train TRAIN_ARGS='--positive-label my_wake_word --epochs 50'"

py-install:
	$(PY) -m pip install -r $(REQ)

record:
	$(PY) $(MODEL_GEN)/record_dataset.py $(RECORD_ARGS)

train:
	$(PY) $(MODEL_GEN)/train_model.py $(TRAIN_ARGS)

train-weights:
	$(PY) $(MODEL_GEN)/train_model.py --no-export $(TRAIN_ARGS)

export-scaffold:
	$(PY) $(MODEL_GEN)/export_model.py

quantize-shrink:
	$(PY) $(MODEL_GEN)/shkrink-mode.py

clean-model-gen:
	rm -f $(MODEL_GEN)/model.onnx $(MODEL_GEN)/model_tiny.onnx \
		$(MODEL_GEN)/wake_word.onnx $(MODEL_GEN)/wake_word_quant.onnx \
		$(MODEL_GEN)/wake_word.pt
