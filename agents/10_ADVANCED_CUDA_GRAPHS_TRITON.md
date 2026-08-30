# 10 — Advanced/Optional: CUDA Graph Capture & Triton Serving

**This entire file is a stretch goal.** The core deliverable (`00`'s Definition of Done) is
already complete after `09`. Do this if there's time/interest left, for extra portfolio depth —
skip straight to `11` otherwise, and say so in the final report rather than leaving it ambiguous.

## A. CUDA Graph capture (kernel-launch-overhead elimination)

A hand pose model at batch size 1 running in a tight per-frame loop is exactly the kind of fixed,
repeated computation graph CUDA Graphs are built for — capture the forward pass once, replay it
many times, skipping Python/CUDA kernel-launch dispatch overhead on every subsequent call.

```python
# optimize/cuda_graph_capture.py
import torch

model.eval()
static_input = torch.zeros(1, 3, 224, 224, device="cuda")

# Warmup on a side stream first — required before capture, per PyTorch's CUDA graph docs
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_output = model(static_input)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay loop — copy new data into static_input's memory, then g.replay()
def infer_via_graph(new_input: torch.Tensor):
    static_input.copy_(new_input)
    g.replay()
    return static_output.clone()
```

**Benchmark specifically the launch-overhead delta**, not just raw latency: compare
`infer_via_graph()` against a normal `model(input)` call, at the *same* batch size and input, using
the same CUDA-event protocol as `06`/`09`. This isolates how much of the win (if any) was
kernel-launch overhead versus how much was already eliminated by TensorRT's own graph fusion in
`09` — it's entirely possible TensorRT already captured most of this gain, in which case a small
or negligible CUDA Graph delta is itself the correct, reportable finding.

**Caveat:** CUDA Graphs require fully static shapes and static memory addresses. If your model has
any dynamic control flow (e.g. a variable number of detected hands before the reconstruction
head), only capture the fixed-shape reconstruction sub-module, not the whole pipeline.

## B. Triton Inference Server (serving-layer literacy demo)

Optional. This step is about demonstrating deployment/serving infrastructure literacy, not about
making the model faster on your laptop — a local Triton instance on a 7GB card is a reasonable
proof-of-concept but not the target production topology.

1. Build a minimal model repository:
   ```
   triton/model_repository/
   └── winner_hand_model/
       ├── config.pbtxt
       └── 1/
           └── model.plan          # your winner_fp16.engine from 09, renamed
   ```
2. `config.pbtxt` (TensorRT backend):
   ```
   name: "winner_hand_model"
   platform: "tensorrt_plan"
   max_batch_size: 1
   input [{ name: "image", data_type: TYPE_FP32, dims: [3, 224, 224] }]
   output [{ name: "joints_3d", data_type: TYPE_FP32, dims: [21, 3] }]
   instance_group [{ kind: KIND_GPU, count: 1 }]
   ```
   (adjust names/dims to match your actual exported graph's I/O)
3. Run the server (Docker required):
   ```bash
   docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
     -v $(pwd)/triton/model_repository:/models \
     nvcr.io/nvidia/tritonserver:24.09-py3 \
     tritonserver --model-repository=/models
   ```
   (pick a Triton container tag that matches a CUDA/driver combo compatible with Blackwell —
   check NVIDIA's container release notes if the above tag predates `sm_120` support.)
4. Send a test request via `tritonclient` (Python) or `curl` against the HTTP endpoint at
   `localhost:8000`, and measure round-trip latency including the serving layer overhead — report
   this separately from the raw in-process TensorRT latency from `09`, since network/serialization
   overhead is a real, distinct cost.

## Definition of Done (only if this file was attempted)

- [ ] CUDA Graph replay latency measured and compared against non-graph TensorRT/PyTorch latency,
      with the delta explicitly attributed to launch overhead (or found negligible, honestly).
- [ ] If Triton was set up: a test client request completes successfully and round-trip latency
      is recorded separately from in-process latency.
- [ ] If this file was skipped: `results/FINAL_REPORT.md` says so explicitly rather than leaving a
      silent gap.
