#!/usr/bin/env python3
"""Path A Stage 2/3 — data plumbing to carry a precomputed reference tensor per example.

Companion to `apply_replay_training_surgery.py` (which unblocked the *model*: fuser keeps
`reference_with_time` in streaming_sum + `lm.py` forward_text applies it over full sequences).
This script wires the *data path* so each replay example can carry its own precomputed
`reference_with_time` tensor from disk all the way into the training forward.

Two files patched (both in the moshi-finetune repo, so idempotent + git-tracked):

1. `finetune/data/interleaver.py`
   - `Sample` gains `reference_tensor` ([T_ref, D] or None).
   - `Batch` gains `reference_tensors` (per-example list or None) and `collate` carries it.
   - `__call__` loads a sibling `<wav-stem>.ref.safetensors` (key "reference") if present.
   Pure-dialogue examples (no sibling file) get None → behave exactly as Stage 1 (no
   conditioning), so a dialogue+replay MIX trains correctly in one run.

2. `train.py` — right after `condition_tensors` is built and before the forward, if the
   batch carries reference tensors, pad them to the batch-max T_ref, build a mask, and inject
   `condition_tensors["reference_with_time"] = ConditionType(ref, mask)`. This is exactly the
   dict the model consumed in the verified forward+backward spike (BACKWARD OK, 438 params
   got grads).

Run under either env (this only touches repo files, not the fork package). Idempotent.
Keeps `.orig` backups.

    python scripts/apply_replay_data_plumbing.py

NOTE (deferred refinements, per the Stage 2/3 design):
- reference is applied at the PREFIX (first T_ref frames) for the first end-to-end run;
  positioning it at the actual ⟨ret⟩ frame is a later correctness pass.
- `first_speaker` prepend is still off (spike config `fuser.prepend=[]`); re-add for the
  real mix run once the reference path is confirmed end-to-end.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERLEAVER = os.path.join(REPO, "finetune", "data", "interleaver.py")
TRAIN = os.path.join(REPO, "train.py")

MARKER = "reference_tensor"  # presence => interleaver already patched


def _backup(path: str, text: str) -> None:
    if not os.path.exists(path + ".orig"):
        open(path + ".orig", "w").write(text)


def patch_interleaver() -> None:
    s = open(INTERLEAVER).read()
    if "reference_tensor: torch.Tensor | None" in s:
        print("interleaver.py already patched.")
        return

    # (a) Sample: add reference_tensor field
    sample_old = (
        "@dataclass\n"
        "class Sample:\n"
        "    codes: torch.Tensor\n"
        "    condition_attributes: ConditionAttributes | None = None\n"
    )
    sample_new = (
        "@dataclass\n"
        "class Sample:\n"
        "    codes: torch.Tensor\n"
        "    condition_attributes: ConditionAttributes | None = None\n"
        "    # [T_ref, D] precomputed reference_with_time (replay data); None for plain dialogue\n"
        "    reference_tensor: torch.Tensor | None = None\n"
    )

    # (b) Batch + collate: carry reference_tensors
    batch_old = (
        "@dataclass\n"
        "class Batch:\n"
        "    codes: torch.Tensor\n"
        "    condition_attributes: list[ConditionAttributes] | None = None\n"
        "\n"
        "    @classmethod\n"
        "    def collate(cls, batch: list[Sample]) -> \"Batch\":\n"
        "        codes = torch.cat([b.codes for b in batch])\n"
        "        if batch[0].condition_attributes is None:\n"
        "            return Batch(codes)\n"
        "        return Batch(codes, [b.condition_attributes for b in batch])\n"
    )
    batch_new = (
        "@dataclass\n"
        "class Batch:\n"
        "    codes: torch.Tensor\n"
        "    condition_attributes: list[ConditionAttributes] | None = None\n"
        "    # per-example [T_ref, D] or None; carries replay reference_with_time\n"
        "    reference_tensors: list | None = None\n"
        "\n"
        "    @classmethod\n"
        "    def collate(cls, batch: list[Sample]) -> \"Batch\":\n"
        "        codes = torch.cat([b.codes for b in batch])\n"
        "        condition_attributes = (\n"
        "            None\n"
        "            if batch[0].condition_attributes is None\n"
        "            else [b.condition_attributes for b in batch]\n"
        "        )\n"
        "        reference_tensors = (\n"
        "            [b.reference_tensor for b in batch]\n"
        "            if any(b.reference_tensor is not None for b in batch)\n"
        "            else None\n"
        "        )\n"
        "        return Batch(codes, condition_attributes, reference_tensors)\n"
    )

    # (c) __call__ return: load sibling reference tensor
    call_old = (
        "            codes = torch.cat([text_tokens, audio_tokens], dim=1)\n"
        "            return Sample(codes, data.get(\"text_conditions\", None))"
    )
    call_new = (
        "            codes = torch.cat([text_tokens, audio_tokens], dim=1)\n"
        "            reference_tensor = _load_reference_tensor(path)\n"
        "            return Sample(codes, data.get(\"text_conditions\", None), reference_tensor)"
    )

    # (d) module-level helper (inserted before the Sample dataclass)
    helper = (
        "def _load_reference_tensor(path):\n"
        "    \"\"\"Load a sibling precomputed reference_with_time tensor for replay examples.\n"
        "    Returns [T_ref, D] or None. File: <wav-stem>.ref.safetensors, key 'reference'.\"\"\"\n"
        "    import os\n"
        "    ref_path = os.path.splitext(str(path))[0] + \".ref.safetensors\"\n"
        "    if not os.path.exists(ref_path):\n"
        "        return None\n"
        "    from safetensors.torch import load_file\n"
        "    return load_file(ref_path)[\"reference\"]\n"
        "\n"
        "\n"
    )

    for name, old in [("Sample", sample_old), ("Batch", batch_old), ("__call__", call_old)]:
        if old not in s:
            raise SystemExit(f"interleaver.py: {name} block not found (version drift?) — patch by hand.")

    _backup(INTERLEAVER, s)
    s = s.replace(sample_old, helper + sample_new)
    s = s.replace(batch_old, batch_new)
    s = s.replace(call_old, call_new)
    open(INTERLEAVER, "w").write(s)
    print(f"patched {INTERLEAVER} (backup at {INTERLEAVER}.orig)")


def patch_train() -> None:
    s = open(TRAIN).read()
    if "condition_tensors[\"reference_with_time\"]" in s:
        print("train.py already patched.")
        return

    old = (
        "            condition_tensors = None\n"
        "            if batch.condition_attributes is not None:\n"
        "                condition_tensors = model.condition_provider.prepare(\n"
        "                    batch.condition_attributes\n"
        "                )\n"
        "\n"
        "            # forward / backward\n"
        "            output = model(codes=codes, condition_tensors=condition_tensors)\n"
    )
    new = (
        "            condition_tensors = None\n"
        "            if batch.condition_attributes is not None:\n"
        "                condition_tensors = model.condition_provider.prepare(\n"
        "                    batch.condition_attributes\n"
        "                )\n"
        "\n"
        "            # Inject precomputed reference_with_time (replay data). condition_provider\n"
        "            # cannot build it (ARC module stripped), so pad per-example tensors to the\n"
        "            # batch-max T_ref and add as a raw ConditionType (matches serve + the\n"
        "            # verified forward+backward spike).\n"
        "            if getattr(batch, \"reference_tensors\", None) is not None:\n"
        "                from moshi.conditioners.base import ConditionType\n"
        "\n"
        "                refs = batch.reference_tensors\n"
        "                present = [r for r in refs if r is not None]\n"
        "                dim = present[0].shape[-1]\n"
        "                T = max(r.shape[0] for r in present)\n"
        "                dt = next(model.parameters()).dtype\n"
        "                ref_cond = torch.zeros(len(refs), T, dim, device=codes.device, dtype=dt)\n"
        "                ref_mask = torch.zeros(len(refs), T, device=codes.device)\n"
        "                for bi, r in enumerate(refs):\n"
        "                    if r is not None:\n"
        "                        L = r.shape[0]\n"
        "                        ref_cond[bi, :L] = r.to(device=codes.device, dtype=dt)\n"
        "                        ref_mask[bi, :L] = 1\n"
        "                if condition_tensors is None:\n"
        "                    condition_tensors = {}\n"
        "                condition_tensors[\"reference_with_time\"] = ConditionType(ref_cond, ref_mask)\n"
        "\n"
        "            # forward / backward\n"
        "            output = model(codes=codes, condition_tensors=condition_tensors)\n"
    )
    if old not in s:
        raise SystemExit("train.py: condition/forward block not found (version drift?) — patch by hand.")
    _backup(TRAIN, s)
    open(TRAIN, "w").write(s.replace(old, new))
    print(f"patched {TRAIN} (backup at {TRAIN}.orig)")


if __name__ == "__main__":
    patch_interleaver()
    patch_train()
    print("\ndata plumbing applied. Next: create a synthetic replay example "
          "(<wav-stem>.ref.safetensors) and run a 1-step training smoke.")
