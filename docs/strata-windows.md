# Strata on native Windows (2× B70)

The launcher runs Strata on Linux only. This page is for running the same engine **natively on Windows 11**,
outside the launcher: three patches for the `hotschmoe/Strata` branch the `qwen38-flashnext-strata` recipe uses, a
tuned engine config, and what was measured. Nothing here changes how the launcher behaves.

- Base: `hotschmoe/Strata` branch `b70-night-20261016` @ `1c360526`.
- Model: ISTA-DASLab GSQ-RCO **IQ3_S** of Qwen3.8-Flash-Next (84 GB, two shards). IQ3_XXS works too (see below).
- Tested on: 4× Arc Pro B70 (two used), EPYC 7402P, Windows 11 Pro 26200, driver 32.0.101.8976, oneAPI 2026.0.
- Full logs and every A/B: [Captain-Tripps/b70-lab, `strata-windows-port/`](https://github.com/Captain-Tripps/b70-lab/tree/main/strata-windows-port).

## Results

Same 7-prompt set, temperature 0, thinking off, answers checked before any speed number.

| Setup | Answers | Decode | Prompt read @ 33.6k |
|---|---|---|---|
| llama.cpp (tuned SYCL fork), **4** cards, UD-Q4_K_XL | correct | 60.6 tok/s | 748 tok/s |
| **Strata, 2 cards, IQ3_S, int8 KV, 1 × 200k, JIT build, `--ple-io mmap`** | **correct** | **~99 tok/s** | **~1,420 tok/s** |

- 112k-token prompt: read in 78 s (1,433 tok/s).
- 2-hour soak through llama-swap: 134 cycles, 0 errors, 0 restarts, decode flat at 93 tok/s; vision 134/134.
- Decode drops with context depth (attention over the history): roughly 35–48 tok/s at 65k–112k tokens of context.

## The patches

Apply in order with `git apply` in the Strata checkout (MIT, derived from Strata; prepared with AI assistance):

| Patch | What |
|---|---|
| [`0001`](strata-windows/0001-sycl-native-Windows-build-icx-of-the-SYCL-engine.patch) native Windows build | CMake flags for `icx` (`/fp:precise`, `/Qmkl:sequential`, C-safe warnings), Level Zero SDK lookup, `ReadFile` positioned reads, a CUDA-only query removed, `<windows.h>` macro clashes (`small`, `IN`, `OUT`, `near`, `far`) undone, `memmem` fallback |
| [`0002`](strata-windows/0002-sycl-Windows-two-card-expert-parallel-through-host-m.patch) two-card expert parallel | Windows Level Zero has no IPC and no peer access between B70s, so all GPUs share **one SYCL context** and the expert-parallel exchange regions are host USM in it. `STRATA_WIN_SHARED_CONTEXT=0` restores one context per device |
| [`0003`](strata-windows/0003-sycl-Windows-JIT-build-option-for-Xe2-extra-AOT-opti.patch) JIT build option | `STRATA_SYCL_XE2_JIT=ON` makes a JIT build define `STRATA_XMX_SG16` like the bmg AOT build; `STRATA_SYCL_AOT_OPTIONS` for extra `ocloc` options |

## Build (JIT)

In a `vcvars64` shell plus the oneAPI 2026.0 component `env\vars.bat` files (tbb, umf, dnnl, mkl, dpl, ocloc,
compiler):

```
cmake -B build-win -S sycl -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_SYCL_AOT= -DSTRATA_SYCL_XE2_JIT=ON ^
      -DSTRATA_SYCL_PARITY=OFF -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx
cmake --build build-win --target strata
```

Put the oneAPI runtime DLLs next to `strata.exe` or on `PATH` (without them the engine exits with 0xC00004BC).

**Use the JIT build.** The oneAPI 2026.0 AOT build for `bmg-g31` miscompiles the batched decode attention over
int8 KV (the MTP verify step): wrong answers, and `qsa_verify_batch_parity` / `kv_hybrid_parity` fail. The same
sources JIT-compiled by the driver pass bitwise. The broken build still looked fast (~97 tok/s) because garbage
output is easy for the draft to predict. Not tested: AOT with oneAPI 2026.1, or the Linux image's compiler.

## Engine config

Template: [`strata-windows/qwen38-flashnext-iq3s-200k.windows.json`](strata-windows/qwen38-flashnext-iq3s-200k.windows.json)
(replace the `<...>` placeholders). The settings that matter:

| Setting | Value | Why |
|---|---|---|
| `--max-context`, `"parallel"` | `200000`, 1 slot | IQ3_S at 2 × 200k (or 2 × 131k) does not fit two 32 GB cards; 2 × 112k does |
| `--kv` | `int8` | needs the JIT build |
| `--prefill` | `8192` | 4096: prompt read 881 tok/s; 10240 puts card 1 at its memory limit and everything runs ~3× slower |
| `--ple-io` | **`mmap`** | see below |
| `--spec` / `--spec-min-p` | `4` / `0` | 3, 5, 6 and p-min 0.1/0.3 were equal or worse |
| `STRATA_NIGHT_PREFILL_GROUPED` | `0` for IQ3_S | the grouped prefill kernel rejects IQ3_S ("card 0 gate/up type 23"); IQ3_XXS needs it on |
| `STRATA_VERIFY_DEVICE_PLAN`, `STRATA_VERIFY_NO_HOST` | `1`, `1` | without them verify waits on a host doorbell and hangs |
| `ONEAPI_DEVICE_SELECTOR` | e.g. `level_zero:2,3` | the two cards to use |

### `--ple-io mmap`, not `direct`

The per-layer embedding table is 27 GB and is read every token. With `--ple-io direct` (unbuffered reads from the
SSD, as in the Linux recipe) decode drifted from ~98 to ~70 tok/s after the drive had taken ~130 GB of new writes.
Heat, clocks, CPU, RAM, the card pair, host-exchange latency (56–64 µs), timer resolution and a reboot were all
ruled out. `mmap` lets the OS cache serve it from RAM: 98–99 tok/s on every start. It needs ~27 GB of free system RAM.
This was only measured on Windows; Linux users of the recipe may want to A/B it.

## IQ3_S vs IQ3_XXS

Perplexity / KL divergence against UD-Q4_K_XL (llama.cpp, 30 × 512-token chunks):

| Quant | Perplexity vs Q4 | Mean KLD | Same top token |
|---|---|---|---|
| IQ3_XXS (76 GB) | +3.5% | 0.211 | 85.3% |
| **IQ3_S (84 GB)** | **+1.0%** | **0.131** | **88.4%** |

IQ3_S is closer to the reference, reads prompts faster in Strata, and decodes ~7–10% slower than IQ3_XXS.

## Serving and vision

Strata's server reads the config: `python sycl\serve\server_intel.py --engine strata --config <CONFIG.json>
--host 127.0.0.1 --port <PORT>`. Under llama-swap on Windows, stop it with `cmdStop: taskkill /T /F /PID ${PID}` so
the engine process ends with the server.

Vision runs on the CPU: `strata-vision.exe` built from llama.cpp's `mtmd` at the commit the engine pins, ISTA-DASLab's
BF16 mmproj, 24 threads, 300 image tokens, about 4.4 s per new image. Add to the config:

```json
"vision": { "exe": "<STRATA_RUNTIME_DIR>\\strata-vision.exe",
            "mmproj": "<MODELS_DIR>\\...\\mmproj-Qwen3.8-Flash-Next-BF16.gguf",
            "model": "<MODELS_DIR>\\...\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
            "gpu": false, "max_tokens": 300, "threads": 24 }
```
