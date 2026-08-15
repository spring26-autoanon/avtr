GCP_PROJECT  ?= adsp-s26-autoanon
ZONE         ?= us-central1-c
INSTANCE     ?= wb-gpu-a1ultra
REMOTE_USER  ?= jupyter
REMOTE_DIR   ?= /home/jupyter/moshirag-evals
LOCAL_DIR    ?= /home/ubuntuvm/Documents/Workspace/Agentic Capstone Playground/moshirag-evals

SSH_ALIAS    ?= $(INSTANCE)

GCLOUD_SSH = gcloud compute ssh $(REMOTE_USER)@$(INSTANCE) \
             --project=$(GCP_PROJECT) --zone=$(ZONE) --tunnel-through-iap
# Non-login SSH shells don't source ~/.profile, so uv (~/.local/bin) isn't on PATH
REMOTE_INIT = export PATH=\$$HOME/.local/bin:\$$PATH

sync:
	rsync -avz \
	  --exclude '.venv' \
	  --exclude '__pycache__' \
	  --exclude 'checkpoint_cache' \
	  --exclude 'evals/results' \
	  --exclude 'demo/sessions' \
	  --exclude 'node_modules' \
	  --exclude 'demo/client/dist' \
	  --exclude '*.pyc' \
	  --exclude '.env' \
	  --exclude '.git' \
	  "$(LOCAL_DIR)/" $(SSH_ALIAS):$(REMOTE_DIR)/

install:
	$(GCLOUD_SSH) -- "$(REMOTE_INIT) && cd $(REMOTE_DIR) && uv sync --all-extras && \
	  uv pip install 'moshi @ git+https://github.com/kyutai-labs/moshi-rag.git#subdirectory=moshi' && \
	  uv pip install 'transformers>=4.57.1,<5' --no-deps && \
	  uv pip install 'tokenizers>=0.22.0,<=0.23.0' --no-deps && \
	  sed -i 's/huggingface-hub>=0.34.0,<1.0/huggingface-hub>=0.34.0/' \
	  .venv/lib/python3*/site-packages/transformers/dependency_versions_table.py"

ssh:
	$(GCLOUD_SSH)

# Run a command on the remote instance, e.g.:  make remote CMD="nvidia-smi"
# If CMD is `uv run ...`, pass --all-extras (see CLAUDE.md) or gpu-extra
# packages like transformers/torch won't be checked/protected by uv's sync.
remote:
	$(GCLOUD_SSH) -- "$(REMOTE_INIT) && cd $(REMOTE_DIR) && $(CMD)"

smoke:
	$(GCLOUD_SSH) -- "$(REMOTE_INIT) && cd $(REMOTE_DIR) && uv run --all-extras evals/runner.py --config configs/baseline_with_retrieval.yaml --mode smoke"

# Launches moshi.server_conditioner + the instrumented moshi.server (see
# scripts/run_demo.sh and scripts/instrumented_server.py) in a detached tmux
# session on the VM and returns immediately — tunnel port 8998 yourself
# afterward: ssh -L 8998:localhost:8998 $(SSH_ALIAS)
demo:
	$(GCLOUD_SSH) -- "$(REMOTE_INIT) && cd $(REMOTE_DIR) && bash scripts/run_demo.sh"

# Pull demo session artifacts back from the VM (server.log, turns.jsonl,
# raw_events.jsonl, step_diag.jsonl, conditioner.log). `demo/sessions` is
# excluded from `sync`'s outbound rsync because the VM is its only writer —
# this is the one-way pull that closes that loop. Client-side captures
# (the "Save audio" .webm and client_audio.json) never reach the VM at all:
# copy those in by hand from the browser machine's downloads folder.
pull-sessions:
	rsync -avz $(SSH_ALIAS):$(REMOTE_DIR)/demo/sessions/ "$(LOCAL_DIR)/demo/sessions/"

# Pull eval results back (see CLAUDE.md's "Code change workflow").
pull-results:
	rsync -avz $(SSH_ALIAS):$(REMOTE_DIR)/evals/results/ "$(LOCAL_DIR)/evals/results/"

.PHONY: sync install ssh remote smoke demo pull-sessions pull-results
