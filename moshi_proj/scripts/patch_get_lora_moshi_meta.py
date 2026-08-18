#!/usr/bin/env python3
"""Fix the fork's get_lora_moshi meta-tensor bug (moshi/models/loaders.py).

`replace_all_linear_with_lora(device=meta)` wraps EVERY linear, creating lora_A/lora_B on the
meta device while reusing the real base weight as `frozen_W`. A finetune adapter (e.g.
checkpoint_700) only covers self_attn + MLP, so after `assign`-loading it, the lora_A/lora_B of
every UNCOVERED linear stay on meta → the following `model.to(device)` raises
"Cannot copy out of meta tensor".

Fix: after loading the adapter, materialize any remaining meta params/buffers as ZEROS. A zero
lora_A/lora_B makes that LoRALinear == frozen_W (identity delta), so uncovered layers are
unchanged and there are no meta tensors left for `.to(device)` to copy.

Run under the FORK env (this edits the installed fork package). Idempotent; keeps loaders.py.orig.

    ~/fork-venv/bin/python scripts/patch_get_lora_moshi_meta.py
"""
import os

import moshi

L = os.path.join(os.path.dirname(moshi.__file__), "models", "loaders.py")

OLD = (
    "        res = model.load_state_dict(lora_state_dict, strict=False, assign=True)\n"
    "        if res.unexpected_keys:\n"
    "            raise RuntimeError(f\"unexpected_keys in the lora weights: {res.unexpected_keys}\")\n"
    "        model = model.to(dtype=dtype, device=device)\n"
)
NEW = (
    "        res = model.load_state_dict(lora_state_dict, strict=False, assign=True)\n"
    "        if res.unexpected_keys:\n"
    "            raise RuntimeError(f\"unexpected_keys in the lora weights: {res.unexpected_keys}\")\n"
    "        # Adapter covers only a subset of linears; lora_A/lora_B of uncovered layers stay on\n"
    "        # meta. Materialize them as zeros (layer == frozen_W) so .to(device) has nothing meta\n"
    "        # to copy. Fixes the get_lora_moshi meta-tensor bug.\n"
    "        for _m in model.modules():\n"
    "            for _pn, _p in list(_m.named_parameters(recurse=False)):\n"
    "                if _p.is_meta:\n"
    "                    setattr(_m, _pn, torch.nn.Parameter(torch.zeros(_p.shape, dtype=dtype, device=device), requires_grad=_p.requires_grad))\n"
    "            for _bn, _b in list(_m.named_buffers(recurse=False)):\n"
    "                if _b is not None and _b.is_meta:\n"
    "                    _m.register_buffer(_bn, torch.zeros(_b.shape, dtype=_b.dtype, device=device))\n"
    "        model = model.to(dtype=dtype, device=device)\n"
)


def main() -> None:
    s = open(L).read()
    if "Fixes the get_lora_moshi meta-tensor bug" in s:
        print("loaders.py already patched.")
        return
    if OLD not in s:
        raise SystemExit("anchor not found in loaders.py (fork version drift?) — patch by hand.")
    if not os.path.exists(L + ".orig"):
        open(L + ".orig", "w").write(s)
    open(L, "w").write(s.replace(OLD, NEW))
    print(f"patched {L} (backup at {L}.orig)")


if __name__ == "__main__":
    main()
