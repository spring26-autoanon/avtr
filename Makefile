GCP_PROJECT  ?= adsp-s26-autoanon
ZONE         ?= us-central1-c
INSTANCE     ?= wb-gpu-a1ultra
REMOTE_USER  ?= jupyter
REMOTE_DIR   ?= /home/jupyter/moshirag-evals
LOCAL_DIR    ?= .

# SSH alias if configured in ~/.ssh/config (optional shorthand)
SSH_ALIAS    ?= $(REMOTE_USER)@$(INSTANCE).$(ZONE).$(GCP_PROJECT)

GCLOUD_SSH = gcloud compute ssh $(REMOTE_USER)@$(INSTANCE) \
             --project=$(GCP_PROJECT) --zone=$(ZONE) --tunnel-through-iap

sync:
	rsync -avz \
	  --exclude '.venv' \
	  --exclude '__pycache__' \
	  --exclude 'checkpoint_cache' \
	  --exclude 'evals/results' \
	  --exclude 'demo/sessions' \
	  --exclude '*.pyc' \
	  --exclude '.env' \
	  --exclude '.git' \
	  $(LOCAL_DIR)/ $(REMOTE_USER)@$(INSTANCE):$(REMOTE_DIR)/

install:
	$(GCLOUD_SSH) -- "cd $(REMOTE_DIR) && uv sync --extra gpu --extra dev && \
	  uv pip install 'moshi @ git+https://github.com/kyutai-labs/moshi-rag.git#subdirectory=moshi'"

ssh:
	$(GCLOUD_SSH)

# Run a command on the remote instance, e.g.:  make remote CMD="nvidia-smi"
remote:
	$(GCLOUD_SSH) -- "cd $(REMOTE_DIR) && $(CMD)"

smoke:
	$(GCLOUD_SSH) -- "cd $(REMOTE_DIR) && uv run evals/runner.py --config configs/baseline_with_retrieval.yaml --mode smoke"

.PHONY: sync install ssh remote smoke
