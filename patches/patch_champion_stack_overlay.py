#!/usr/bin/env python3
"""Self-contained champion overlay: GDN v5 + optional draft INT4 S/M1."""
from __future__ import annotations

import os
import sys

# ===== GDN v5 =====
#!/usr/bin/env python3
"""XPU GDN mixed spec + non-spec split (v5) — causal_conv1d root cause.

The fused ``torch.ops._xpu_C.gdn_attention`` host binding refuses a batch
that mixes spec-decode tokens with non-spec (prefill + decode) tokens:

    causal_conv1d does not support spec-decode and non-spec (prefill +
    decode) tokens in the same invocation

That is a host ``TORCH_CHECK``, not a missing SYCL kernel. The two launchers
already exist; they cannot share one invocation because Xe2 non-spec
intermediates are chunk-padded and spec intermediates are not, and the fused
wrapper consumes exactly one 5-tensor group.

CUDA already splits this in Python (``QwenGDNLinearAttention._forward_core``).
This patch does the same at the XPU fused-op boundary:

  * homogeneous batch → one fused call (unchanged C1 / no-spec path)
  * mixed batch → compact each group with ``index_select``, one fused call
    per group with ``num_actual_tokens = group_count`` and ``token_indx =
    arange``, then ``index_copy_`` both ``core_attn_out`` **and** ``z``

v1–v4 failed the C++ size-sum / ``narrow`` contract (global indices into a
compact buffer, or unused-side dummy tensors that still counted as tokens).
v5 uses ``None`` for the idle optional side (binding is ``Tensor?``).

Applies to the import path the f01e24f6 image actually uses
(``/workspace/vllm/vllm/_xpu_ops.py``), then site-packages as fallback.
Idempotent (``B70_GDN_MIXED_SPLIT_V5``).
"""

import sys
from pathlib import Path

MARKER = "B70_GDN_MIXED_SPLIT_V5"

CANDIDATES = (
    Path("/workspace/vllm/vllm/_xpu_ops.py"),
    Path("/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"),
    Path("/opt/vllm/vllm/_xpu_ops.py"),
)

OLD = """    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
        z,
        projected_states_qkvz,
        projected_states_ba,
        self.num_k_heads,
        self.num_v_heads,
        self.head_k_dim,
        self.head_v_dim,
        conv_state=self.kv_cache[0],
        ssm_state=self.kv_cache[1],
        conv_weights=conv_weights,
        conv_bias=self.conv1d.bias,
        activation=self.activation,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        num_prefills=num_prefills,  # type: ignore[attr-defined]
        num_decodes=num_decodes,  # type: ignore[attr-defined]
        num_spec_decodes=num_spec_decodes,  # type: ignore[attr-defined]
        has_initial_state=has_initial_state,  # type: ignore[attr-defined]
        non_spec_query_start_loc=non_spec_query_start_loc,  # type: ignore[attr-defined]
        non_spec_token_indx=non_spec_token_indx,  # type: ignore[attr-defined]
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,  # type: ignore[attr-defined]
        spec_query_start_loc=spec_query_start_loc,  # type: ignore[attr-defined]
        spec_token_indx=spec_token_indx,  # type: ignore[attr-defined]
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,  # type: ignore[attr-defined]
        num_actual_tokens=num_actual_tokens,  # type: ignore[attr-defined]
        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
    )
"""

NEW = '''    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )

    # B70_GDN_MIXED_SPLIT_V5: fused XPU causal_conv1d is exclusive
    # (spec XOR non-spec). Compact + two fused calls + scatter z/out.
    # Homogeneous batches keep the single call (C1 MTP / no-spec).
    _mixed = (
        num_spec_decodes > 0
        and (num_prefills + num_decodes) > 0
        and non_spec_token_indx is not None
        and spec_token_indx is not None
    )
    _reorder = not self.gqa_interleaved_layout

    def _invoke(
        _out,
        _z,
        _qkvz,
        _ba,
        *,
        n_pref,
        n_dec,
        n_spec,
        n_tok,
        has_init,
        ns_loc,
        ns_idx,
        ns_state,
        sp_loc,
        sp_idx,
        sp_state,
        n_acc,
    ):
        torch.ops._xpu_C.gdn_attention(
            _out,
            _z,
            _qkvz,
            _ba,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            conv_state=self.kv_cache[0],
            ssm_state=self.kv_cache[1],
            conv_weights=conv_weights,
            conv_bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_prefills=n_pref,
            num_decodes=n_dec,
            num_spec_decodes=n_spec,
            has_initial_state=has_init,
            non_spec_query_start_loc=ns_loc,
            non_spec_token_indx=ns_idx,
            non_spec_state_indices_tensor=ns_state,
            spec_query_start_loc=sp_loc,
            spec_token_indx=sp_idx,
            spec_state_indices_tensor=sp_state,
            num_accepted_tokens=n_acc,
            num_actual_tokens=n_tok,
            tp_size=self.tp_size,
            reorder_input=_reorder,
        )

    if not _mixed:
        _invoke(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            n_pref=num_prefills,
            n_dec=num_decodes,
            n_spec=num_spec_decodes,
            n_tok=num_actual_tokens,
            has_init=has_initial_state,
            ns_loc=non_spec_query_start_loc,
            ns_idx=non_spec_token_indx,
            ns_state=non_spec_state_indices_tensor,
            sp_loc=spec_query_start_loc,
            sp_idx=spec_token_indx,
            sp_state=spec_state_indices_tensor,
            n_acc=num_accepted_tokens,
        )
        return

    def _i32(t):
        if t is None:
            return None
        if t.dtype != torch.int32:
            t = t.to(torch.int32)
        return t.contiguous()

    _nsti = _i32(non_spec_token_indx)
    _sti = _i32(spec_token_indx)
    _n_ns = int(_nsti.numel())
    _n_sp = int(_sti.numel())
    _dev = core_attn_out.device
    if not getattr(self, "_b70_gdn_split_logged", False):
        logger.info(
            "[B70] GDN mixed split prefill=%d decode=%d spec=%d n_ns=%d n_sp=%d",
            num_prefills,
            num_decodes,
            num_spec_decodes,
            _n_ns,
            _n_sp,
        )
        self._b70_gdn_split_logged = True

    def _compact(src, idx):
        return src.index_select(0, idx.to(torch.long)).contiguous()

    if _n_ns > 0:
        _z_ns = _compact(z, _nsti)
        _qkvz_ns = _compact(projected_states_qkvz, _nsti)
        _ba_ns = _compact(projected_states_ba, _nsti)
        _out_ns = core_attn_out.new_empty((_n_ns,) + tuple(core_attn_out.shape[1:]))
        _ar_ns = torch.arange(_n_ns, dtype=torch.int32, device=_dev)
        _invoke(
            _out_ns,
            _z_ns,
            _qkvz_ns,
            _ba_ns,
            n_pref=num_prefills,
            n_dec=num_decodes,
            n_spec=0,
            n_tok=_n_ns,
            has_init=has_initial_state,
            ns_loc=non_spec_query_start_loc,
            ns_idx=_ar_ns,
            ns_state=non_spec_state_indices_tensor,
            sp_loc=None,
            sp_idx=None,
            sp_state=None,
            n_acc=None,
        )
        core_attn_out.index_copy_(0, _nsti.to(torch.long), _out_ns)
        z.index_copy_(0, _nsti.to(torch.long), _z_ns)

    if _n_sp > 0:
        _z_sp = _compact(z, _sti)
        _qkvz_sp = _compact(projected_states_qkvz, _sti)
        _ba_sp = _compact(projected_states_ba, _sti)
        _out_sp = core_attn_out.new_empty((_n_sp,) + tuple(core_attn_out.shape[1:]))
        _ar_sp = torch.arange(_n_sp, dtype=torch.int32, device=_dev)
        _invoke(
            _out_sp,
            _z_sp,
            _qkvz_sp,
            _ba_sp,
            n_pref=0,
            n_dec=0,
            n_spec=num_spec_decodes,
            n_tok=_n_sp,
            has_init=None,
            ns_loc=None,
            ns_idx=None,
            ns_state=None,
            sp_loc=spec_query_start_loc,
            sp_idx=_ar_sp,
            sp_state=spec_state_indices_tensor,
            n_acc=num_accepted_tokens,
        )
        core_attn_out.index_copy_(0, _sti.to(torch.long), _out_sp)
        z.index_copy_(0, _sti.to(torch.long), _z_sp)
'''


def resolve_paths() -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for p in CANDIDATES:
        if p.is_file():
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                found.append(rp)
    try:
        import vllm

        p = (Path(vllm.__file__).resolve().parent / "_xpu_ops.py")
        if p.is_file() and p not in seen:
            found.append(p)
    except Exception:
        pass
    if not found:
        sys.exit("vllm/_xpu_ops.py not found")
    return found


def patch(path: Path | None = None) -> Path:
    targets = [path] if path is not None else resolve_paths()
    last = targets[-1]
    patched = 0
    for p in targets:
        text = p.read_text()
        if MARKER in text:
            print(f"[gdn-split-v5] already patched {p}")
            last = p
            continue
        if OLD not in text:
            sys.exit(f"[gdn-split-v5] anchor not found in {p}")
        p.write_text(text.replace(OLD, NEW, 1))
        print(f"[gdn-split-v5] patched {p}")
        last = p
        patched += 1
    if path is None:
        print(f"[gdn-split-v5] applied to {len(targets)} file(s), new={patched}")
    return last




_DRAFT_S = '#!/usr/bin/env python3\n"""B70 Fase S: draft MTP LM head cuantizado a INT4 g128 sym (GPTQ format).\n\nEl LM head fp16 compartido (5120x248320 = 2.54 GB) se lee 5x/paso MTP4\n(4 drafts + 1 target) = 12.7 GB/step a ~98% del pico de DRAM (21.3 ms,\n41% del tiempo de GEMM). Cuantizar SOLO la copia del draft a INT4 g128 sym\n(0.66 GB/paso incl. scales) reduce las 4 pasadas del draft a ~4x0.66 GB\n→ -7.6 GB/step ≈ -13 ms/step → objetivo ~108 tok/s.\n\nLossless: el draft MTP usa SU PROPIO lm_head (DraftModelProposer._maybe_share\n_lm_head es no-op — no comparte el del target). Cuantizar la copia del draft\nno toca el target de verificacion (fp16) y los tokens del draft se verifican\ncontra el target: la secuencia emitida es identica a MTP4 baseline (greedy).\n\nFormato del peso INT4 = exactamente el que consume la op YA existente\n``torch.ops._xpu_C.int4_gemm_w4a16`` (el que usan los layers INC/GPTQ del\ncuerpo): qweight int32 [K/8, N] NT (nibbles secuenciales LSB-first,\nvalor almacenado = q + 8), scales fp16 [K/128, N], qzeros int8 [1] = 8.\n\nEnv-gated: B70_DRAFT_LMHEAD_INT4=1 (default off = comportamiento identico).\nAnclas verificadas contra la imagen vllm/vllm-openai-xpu@2c427ef (vLLM\n0.26.1.dev457.gc810e5ee9).\n"""\nfrom __future__ import annotations\n\nimport os\nimport sys\n\nMARKER = "B70_DRAFT_LMHEAD_INT4"\n\nHELPER_MODULE = "b70_draft_lmhead_int4.py"\n\nHELPER_SOURCE = \'\'\'\\\n"""B70 Fase S runtime helper: draft MTP LM head INT4 g128 sym.\n\nEscribido por patch_draft_lmhead_int4.py dentro del contenedor. Provee la\ncuantizacion one-time del lm_head fp16 compartido a GPTQ INT4 g128 sym y el\nruteo de las 4 pasadas del draft por ``int4_gemm_w4a16``. El target queda\nfp16 (lossless).\n"""\nfrom __future__ import annotations\n\nimport os\n\nimport torch\n\n\ndef quantize_lmhead_to_int4(weight: torch.Tensor, group_size: int = 128):\n    """Quantiza un lm_head fp16 [N, K] a GPTQ INT4 g128 sym.\n\n    Returns (qweight, scales, qzeros, group_size):\n      qweight: int32 [K//8, N] en layout NT (strides[-2] == 1), nibbles\n               secuenciales LSB-first, valor almacenado = q + 8 (q in [-8, 7])\n      scales:  fp16 [K//group_size, N]\n      qzeros:  int8 tensor([8])  -> rama simetrica de int4_gemm_w4a16\n    """\n    device = weight.device\n    N, K = weight.shape\n    num_groups = K // group_size\n    chunk = 4096\n    shifts = torch.tensor(\n        [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device\n    )\n    parts = []\n    scale_parts = []\n    for i in range(0, N, chunk):\n        wc = weight[i : i + chunk].float()  # [c, K] fp32 (chunked: no 5 GB temp)\n        wg = wc.view(wc.shape[0], num_groups, group_size)\n        maxabs = wg.abs().amax(dim=-1)  # [c, g]\n        scale = maxabs / 7.0\n        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)\n        stored = q + 8  # 0..15\n        qv = stored.view(wc.shape[0], num_groups, group_size // 8, 8)\n        packed = (qv << shifts).sum(dim=-1).to(torch.int32).reshape(\n            wc.shape[0], K // 8\n        )\n        parts.append(packed)\n        scale_parts.append(scale.half())\n    qweight_contig = torch.cat(parts, dim=0)  # [N, K//8] int32\n    scales_contig = torch.cat(scale_parts, dim=0)  # [N, g] fp16\n    # Layout NT requerido por la op (strides[-2] == 1) + scales contiguas\n    qweight = qweight_contig.t()  # [K//8, N], strides (1, K//8)\n    scales = scales_contig.t().contiguous()  # [g, N]\n    qzeros = torch.tensor([8], dtype=torch.int8, device=device)\n    return qweight, scales, qzeros, group_size\n\n\ndef int4_lmhead_logits(\n    x: torch.Tensor,\n    qweight: torch.Tensor,\n    scales: torch.Tensor,\n    qzeros: torch.Tensor,\n    group_size: int,\n) -> torch.Tensor:\n    """Logits [.., vocab] via int4_gemm_w4a16 (mismo formato que el cuerpo)."""\n    flat = x.reshape(-1, x.shape[-1])\n    logits = torch.ops._xpu_C.int4_gemm_w4a16(\n        flat, qweight, None, scales, qzeros, group_size, None\n    )\n    return logits.reshape(*x.shape[:-1], qweight.shape[1])\n\n\n@torch.no_grad()\ndef build_draft_lmhead_int4(model) -> None:\n    """Cuantiza el lm_head fp16 compartido del draft (one-time, no-op si no\n    hay env gate o si ya se construyo). Almacena en model._b70_lmhead_int4."""\n    if os.environ.get("B70_DRAFT_LMHEAD_INT4") != "1":\n        return\n    if getattr(model, "_b70_lmhead_int4", None) is not None:\n        return\n    head = getattr(model, "lm_head", None)\n    weight = getattr(head, "weight", None)\n    if weight is None:\n        print("[B70] draft LM head INT4: lm_head.weight no disponible; "\n              "draft sigue por fp16", flush=True)\n        return\n    print("[B70] draft LM head INT4: cuantizando lm_head fp16 "\n          f"{tuple(weight.shape)} -> INT4 g128 sym (one-time)", flush=True)\n    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(\n        weight.detach()\n    )\n    model._b70_lmhead_int4 = (qweight, scales, qzeros, group_size)\n    fp16_bytes = weight.numel() * weight.element_size()\n    int4_bytes = qweight.numel() * qweight.element_size() + (\n        scales.numel() * scales.element_size()\n    )\n    print(f"[B70] draft LM head INT4: listo. {fp16_bytes/1e9:.2f} GB fp16 -> "\n          f"{int4_bytes/1e9:.2f} GB INT4 (ahorro "\n          f"{(fp16_bytes - int4_bytes)/1e6:.1f} MB/lectura)", flush=True)\n\n\ndef draft_lmhead_int4_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:\n    """Logits del draft via la copia INT4 (4 pasadas/paso -> 0.66 GB c/u)."""\n    qweight, scales, qzeros, group_size = model._b70_lmhead_int4\n    logits = int4_lmhead_logits(\n        hidden_states, qweight, scales, qzeros, group_size\n    )\n    org = getattr(getattr(model, "logits_processor", None), "org_vocab_size", None)\n    if org is not None and logits.shape[-1] > org:\n        logits = logits[..., :org]\n    return logits\n\'\'\'\n\nQWMTP_OLD = (\n    "    def compute_logits(\\n"\n    "        self,\\n"\n    "        hidden_states: torch.Tensor,\\n"\n    "        spec_step_idx: int = 0,\\n"\n    "    ) -> torch.Tensor | None:\\n"\n    "        return self.logits_processor(self.lm_head, hidden_states)\\n"\n)\nQWMTP_NEW = (\n    "    def compute_logits(\\n"\n    "        self,\\n"\n    "        hidden_states: torch.Tensor,\\n"\n    "        spec_step_idx: int = 0,\\n"\n    "    ) -> torch.Tensor | None:\\n"\n    "        if os.environ.get(\\"B70_DRAFT_LMHEAD_INT4\\") == \\"1\\":\\n"\n    "            from vllm.model_executor.models.b70_draft_lmhead_int4 import (\\n"\n    "                build_draft_lmhead_int4,\\n"\n    "                draft_lmhead_int4_logits,\\n"\n    "            )\\n"\n    "\\n"\n    "            if getattr(self, \\"_b70_lmhead_int4\\", None) is None:\\n"\n    "                build_draft_lmhead_int4(self)\\n"\n    "            if getattr(self, \\"_b70_lmhead_int4\\", None) is not None:\\n"\n    "                return draft_lmhead_int4_logits(self, hidden_states)\\n"\n    "        return self.logits_processor(self.lm_head, hidden_states)\\n"\n)\n\n\ndef _write_helper(vllm_dir: str) -> str:\n    models_dir = os.path.join(vllm_dir, "model_executor", "models")\n    os.makedirs(models_dir, exist_ok=True)\n    path = os.path.join(models_dir, HELPER_MODULE)\n    existing = None\n    if os.path.exists(path):\n        existing = open(path).read()\n    if existing == HELPER_SOURCE:\n        print(f"helper already present {path}")\n        return path\n    with open(path, "w") as f:\n        f.write(HELPER_SOURCE)\n    print(f"helper written {path}")\n    return path\n\n\ndef _patch_qwen3_5_mtp(vllm_dir: str) -> None:\n    path = os.path.join(vllm_dir, "model_executor", "models", "qwen3_5_mtp.py")\n    text = open(path).read()\n    if MARKER in text:\n        print(f"already patched {path}")\n        return\n    if QWMTP_OLD not in text:\n        sys.exit(f"anchor not found in {path}: compute_logits")\n    text = text.replace(QWMTP_OLD, QWMTP_NEW, 1)\n    if "\\nimport os\\n" not in text and not text.startswith("import os\\n"):\n        text = text.replace("import torch\\n", "import os\\nimport torch\\n", 1)\n        if "\\nimport os\\n" not in text and not text.startswith("import os\\n"):\n            sys.exit(f"could not inject import os in {path}")\n    compile(text, path, "exec")\n    open(path, "w").write(text)\n    print(f"patched {path}")\n\n\ndef main() -> None:\n    import vllm\n\n    vllm_dir = os.path.dirname(vllm.__file__)\n    _write_helper(vllm_dir)\n    _patch_qwen3_5_mtp(vllm_dir)\n\n\nif __name__ == "__main__":\n    main()\n'
_DRAFT_M = '#!/usr/bin/env python3\n"""B70 Fase M1: MTP layer (draft) cuantizado a INT4 g128 sym (formato GPTQ).\n\nEl modulo MTP del draft (Qwen3_5MultiTokenPredictor: fc + full_attention\ndecoder layer) esta en BF16 (forzado por B70_MTP_BF16_DRAFT=1) y se lee\n~0.85 GB/paso (4 pasadas/paso = 3.4 GB). Cuantizar sus 5 linears\n(fc, qkv_proj, o_proj, gate_up_proj, down_proj) a INT4 g128 sym\n(~0.21 GB incl. scales) reusa el kernel YA existente\n``torch.ops._xpu_C.int4_gemm_w4a16`` (formato GPTQ del cuerpo) y elimina\n~2.6 GB/paso de lecturas de DRAM.\n\nLossless: el target de verificacion no se toca; los tokens del draft se\nverifican contra el target (greedy) -> la secuencia emitida es identica.\nGATE de aceptacion: el draft INT4 puede proponer distinto; si la aceptacion\nbaja >0.03 la ganancia de bytes puede perderse -> medir ANTES vs DESPUES.\n\nMecanismo: los linears del MTP tienen ``quant_method`` (UnquantizedLinearMethod)\ncuyo ``apply(layer, x, bias)`` hace el matmul. Se intercambia ``quant_method``\npor un metodo duck-typed que rutea por int4_gemm_w4a16 con la copia INT4.\nEl resto del layer (attention, rope, norms, silu) queda intacto.\n\nIMPORTANTE (compilacion): el hook va en ``Qwen3_5MTP.load_weights`` (eager,\nal cargar el modelo) y NO en forward: el forward de Qwen3_5MTP esta\ndecorado con @support_torch_compile (AOT fullgraph) y una construccion\nlazy en forward rompe el trace de dynamo ("Failed to trace builtin operator\nprint" en warmup). La construccion en load_weights corre antes de cualquier\ntrace -> el grafo compilado ya ve los quant_method INT4.\n\nEnv-gated: B70_DRAFT_MTP_INT4=1 (default off = comportamiento identico).\nAnclas verificadas contra la imagen vllm/vllm-openai-xpu@2c427ef (vLLM\n0.26.1.dev457.gc810e5ee9).\n"""\nfrom __future__ import annotations\n\nimport os\nimport sys\n\nMARKER = "B70_DRAFT_MTP_INT4"\n\nHELPER_MODULE = "b70_draft_mtp_int4.py"\n\nHELPER_SOURCE = \'\'\'\\\n"""B70 Fase M1 runtime helper: draft MTP layer INT4 g128 sym.\n\nEscribido por patch_draft_mtp_int4.py dentro del contenedor. Provee la\ncuantizacion one-time de los 5 linears del modulo MTP del draft a GPTQ\nINT4 g128 sym y el ruteo por ``int4_gemm_w4a16``. El target queda intacto.\n"""\nfrom __future__ import annotations\n\nimport os\n\nimport torch\n\n\ndef quantize_to_int4(weight: torch.Tensor, group_size: int = 128):\n    """Quantiza un peso [N, K] a GPTQ INT4 g128 sym (formato int4_gemm_w4a16).\n\n    Returns (qweight, scales, qzeros, group_size):\n      qweight: int32 [K//8, N] layout NT (strides[-2] == 1), nibbles\n               secuenciales LSB-first, valor almacenado = q + 8 (q in [-8, 7])\n      scales:  fp16 [K//group_size, N]\n      qzeros:  int8 tensor([8])  -> rama simetrica de int4_gemm_w4a16\n    """\n    device = weight.device\n    N, K = weight.shape\n    num_groups = K // group_size\n    chunk = 4096\n    shifts = torch.tensor(\n        [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device\n    )\n    parts = []\n    scale_parts = []\n    for i in range(0, N, chunk):\n        wc = weight[i : i + chunk].float()\n        wg = wc.view(wc.shape[0], num_groups, group_size)\n        maxabs = wg.abs().amax(dim=-1)\n        scale = maxabs / 7.0\n        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)\n        stored = q + 8\n        qv = stored.view(wc.shape[0], num_groups, group_size // 8, 8)\n        packed = (qv << shifts).sum(dim=-1).to(torch.int32).reshape(\n            wc.shape[0], K // 8\n        )\n        parts.append(packed)\n        scale_parts.append(scale.half())\n    qweight_contig = torch.cat(parts, dim=0)\n    scales_contig = torch.cat(scale_parts, dim=0)\n    qweight = qweight_contig.t()\n    scales = scales_contig.t().contiguous()\n    qzeros = torch.tensor([8], dtype=torch.int8, device=device)\n    return qweight, scales, qzeros, group_size\n\n\ndef _collect_linears(predictor) -> list[tuple[str, torch.nn.Module]]:\n    """Lista (name, linear) de los 5 linears del MTP predictor a INT4."""\n    found: list[tuple[str, torch.nn.Module]] = []\n    found.append(("fc", predictor.fc))\n    for li, layer in enumerate(predictor.layers):\n        attn = getattr(layer, "self_attn", None)\n        if attn is not None:\n            found.append((f"layers.{li}.self_attn.qkv_proj", attn.qkv_proj))\n            found.append((f"layers.{li}.self_attn.o_proj", attn.o_proj))\n        mlp = getattr(layer, "mlp", None)\n        if mlp is not None:\n            gate_up = getattr(mlp, "gate_up_proj", None)\n            down = getattr(mlp, "down_proj", None)\n            if gate_up is not None:\n                found.append((f"layers.{li}.mlp.gate_up_proj", gate_up))\n            if down is not None:\n                found.append((f"layers.{li}.mlp.down_proj", down))\n    return found\n\n\nclass _B70MTPInt4LinearMethod:\n    """Duck-typed quant_method: apply() rutea por int4_gemm_w4a16."""\n\n    def __init__(self, qweight, scales, qzeros, group_size):\n        self.qweight = qweight\n        self.scales = scales\n        self.qzeros = qzeros\n        self.group_size = group_size\n\n    def create_weights(self, *args, **kwargs):\n        pass\n\n    def process_weights_after_loading(self, *args, **kwargs):\n        pass\n\n    def apply(self, layer, x, bias):\n        flat = x.reshape(-1, x.shape[-1])\n        if flat.dtype != torch.float16:\n            flat = flat.to(torch.float16)\n        out = torch.ops._xpu_C.int4_gemm_w4a16(\n            flat, self.qweight, None, self.scales, self.qzeros,\n            self.group_size, None,\n        )\n        return out.reshape(*x.shape[:-1], self.qweight.shape[1])\n\n\n@torch.no_grad()\ndef build_draft_mtp_int4(model) -> None:\n    """Cuantiza los linears del MTP predictor (one-time en load_weights; no-op\n    si no hay env gate o si ya se construyo). Almacena en\n    model._b70_mtp_int4_built."""\n    if os.environ.get("B70_DRAFT_MTP_INT4") != "1":\n        return\n    if getattr(model, "_b70_mtp_int4_built", False):\n        return\n    predictor = getattr(model, "model", None)\n    if predictor is None or not hasattr(predictor, "layers"):\n        print("[B70] draft MTP INT4: no Qwen3_5MultiTokenPredictor; "\n              "MTP queda en BF16", flush=True)\n        return\n    linears = _collect_linears(predictor)\n    if not linears:\n        print("[B70] draft MTP INT4: no linears encontrados; skip", flush=True)\n        return\n    print(f"[B70] draft MTP INT4: cuantizando {len(linears)} linears "\n          f"del MTP -> INT4 g128 sym (one-time)", flush=True)\n    total_fp16 = 0\n    total_int4 = 0\n    for name, lin in linears:\n        w = getattr(lin, "weight", None)\n        if w is None:\n            continue\n        orig_shape = tuple(w.shape)\n        qweight, scales, qzeros, gs = quantize_to_int4(w.detach())\n        lin._b70_mtp_int4 = _B70MTPInt4LinearMethod(\n            qweight, scales, qzeros, gs\n        )\n        lin.quant_method = lin._b70_mtp_int4\n        fp16_bytes = w.numel() * w.element_size()\n        int4_bytes = qweight.numel() * qweight.element_size() + (\n            scales.numel() * scales.element_size()\n        )\n        total_fp16 += fp16_bytes\n        total_int4 += int4_bytes\n        with torch.no_grad():\n            lin.weight.set_(\n                torch.empty(0, dtype=w.dtype, device=w.device)\n            )\n        print(f"[B70] draft MTP INT4: {name} {orig_shape} "\n              f"{fp16_bytes/1e6:.0f} MB -> {int4_bytes/1e6:.0f} MB "\n              f"(fp16 liberado)", flush=True)\n    model._b70_mtp_int4_built = True\n    print(f"[B70] draft MTP INT4: listo. {total_fp16/1e9:.2f} GB BF16 -> "\n          f"{total_int4/1e9:.2f} GB INT4 (ahorro "\n          f"{(total_fp16 - total_int4)/1e6:.0f} MB/lectura)", flush=True)\n\'\'\'\n\nQWMTP_FORWARD_OLD = (\n    "        loader = AutoWeightsLoader(self)\\n"\n    "        return loader.load_weights(remap_weight_names(weights))\\n"\n)\nQWMTP_FORWARD_NEW = (\n    "        loader = AutoWeightsLoader(self)\\n"\n    "        result = loader.load_weights(remap_weight_names(weights))\\n"\n    "        if os.environ.get(\\"B70_DRAFT_MTP_INT4\\") == \\"1\\":\\n"\n    "            from vllm.model_executor.models.b70_draft_mtp_int4 import (\\n"\n    "                build_draft_mtp_int4,\\n"\n    "            )\\n"\n    "\\n"\n    "            build_draft_mtp_int4(self)\\n"\n    "        return result\\n"\n)\n\n\ndef _write_helper(vllm_dir: str) -> str:\n    models_dir = os.path.join(vllm_dir, "model_executor", "models")\n    os.makedirs(models_dir, exist_ok=True)\n    path = os.path.join(models_dir, HELPER_MODULE)\n    existing = None\n    if os.path.exists(path):\n        existing = open(path).read()\n    if existing == HELPER_SOURCE:\n        print(f"helper already present {path}")\n        return path\n    with open(path, "w") as f:\n        f.write(HELPER_SOURCE)\n    print(f"helper written {path}")\n    return path\n\n\ndef _patch_qwen3_5_mtp(vllm_dir: str) -> None:\n    path = os.path.join(vllm_dir, "model_executor", "models", "qwen3_5_mtp.py")\n    text = open(path).read()\n    if MARKER in text:\n        print(f"already patched {path}")\n        return\n    if QWMTP_FORWARD_OLD not in text:\n        sys.exit(f"anchor not found in {path}: Qwen3_5MTP.forward")\n    text = text.replace(QWMTP_FORWARD_OLD, QWMTP_FORWARD_NEW, 1)\n    if "\\nimport os\\n" not in text and not text.startswith("import os\\n"):\n        text = text.replace("import torch\\n", "import os\\nimport torch\\n", 1)\n        if "\\nimport os\\n" not in text and not text.startswith("import os\\n"):\n            sys.exit(f"could not inject import os in {path}")\n    compile(text, path, "exec")\n    open(path, "w").write(text)\n    print(f"patched {path}")\n\n\ndef main() -> None:\n    import vllm\n\n    vllm_dir = os.path.dirname(vllm.__file__)\n    _write_helper(vllm_dir)\n    _patch_qwen3_5_mtp(vllm_dir)\n\n\nif __name__ == "__main__":\n    main()\n'

def _run_src(name, src):
    g = {"__name__": "__main__", "__file__": name}
    exec(compile(src, name, "exec"), g)

def main():
    patch()
    if os.environ.get("B70_DRAFT_LMHEAD_INT4") == "1":
        print("[stack] draft S")
        _run_src("patch_draft_lmhead_int4.py", _DRAFT_S)
    if os.environ.get("B70_DRAFT_MTP_INT4") == "1":
        print("[stack] draft M1")
        _run_src("patch_draft_mtp_int4.py", _DRAFT_M)

if __name__ == "__main__":
    main()
