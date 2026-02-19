```markdown
# Mini ML Inference Engine (C++): Build Plan + Spec (VS Code Friendly)

A hands-on, portfolio-grade project that demonstrates **ML inference fundamentals + systems/performance engineering**:
- **Tensors + ops**
- **Computation graph**
- **Runtime executor**
- **Optimizations (fusion, memory reuse, threading)**
- **Benchmarks + tests**
- (Optional later) **CUDA acceleration** for matmul/ReLU

This README is designed so you can **build it in VS Code** using **CMake** on macOS/Linux/Windows.

---

## 0) What You’re Building

### High-level goal
Implement a small inference runtime that can execute a feed-forward neural network defined as a computation graph:

```

Input -> Linear -> ReLU -> Linear -> Softmax -> Output

```

### What “inference engine” means here
- Takes model weights + graph definition
- Runs **forward pass only**
- Produces outputs
- Measures latency/throughput
- Applies basic graph/runtime optimizations

---

## 1) Project Requirements

### Language/Tools
- C++17 or newer
- CMake >= 3.20
- VS Code + extensions:
  - **CMake Tools** (ms-vscode.cmake-tools)
  - **C/C++** (ms-vscode.cpptools)

### Optional
- OpenMP (easy multithreading)
- GoogleTest (unit tests)
- `nlohmann/json` (simple model format)

---

## 2) Milestone Roadmap (Do this in order)

### Milestone A — Core tensor + CPU ops (foundation)
✅ Tensor class (shape, dtype, storage)  
✅ Basic ops:
- MatMul (2D)
- Add (bias)
- ReLU
- Softmax
- (Nice) LayerNorm / GELU later

**Exit criteria**: You can run a small “2-layer MLP” and match a reference result.

---

### Milestone B — Computation graph + executor
✅ Graph representation:
- Nodes (op type + attributes)
- Edges (tensor references)
- Topological order

✅ Executor:
- Runs nodes in order
- Manages intermediate tensors

**Exit criteria**: You can define a model in a file and execute it.

---

### Milestone C — Optimization passes (what makes it impressive)
Implement at least **two**:

1) **Operator fusion**
- Fuse `Linear + ReLU` into one op to reduce memory traffic
2) **Memory planning / reuse**
- Reuse buffers for intermediates (simple liveness-based reuse)
3) **Constant folding** (optional)
- Precompute constants at “compile” time
4) **Multithreading**
- OpenMP or std::thread in MatMul

**Exit criteria**: You can show measurable speedup (e.g., 1.5×–3× on CPU).

---

### Milestone D — Benchmarking + tests
✅ Microbenchmarks:
- MatMul sizes (e.g., 256, 512, 1024)
- End-to-end model latency

✅ Unit tests:
- Tensor shape rules
- ReLU/Softmax correctness
- MatMul correctness vs reference

**Exit criteria**: One command runs tests + benchmark.

---

## 3) Deliverables (What Goes On Your Resume)

By the end you should be able to say:
- Built a C++ inference runtime supporting computation graphs and tensor ops
- Implemented graph optimizations (fusion + memory reuse)
- Achieved X× speedup vs baseline
- Added benchmarking + correctness tests

---

## 4) Repository Layout (Recommended)

```

mini-infer/
CMakeLists.txt
README.md
.vscode/
settings.json
launch.json
tasks.json (optional)
include/
miniinfer/
tensor.h
dtype.h
shape.h
op.h
graph.h
executor.h
model.h
profiler.h
optimizer.h
src/
tensor.cpp
ops/
matmul.cpp
relu.cpp
softmax.cpp
add.cpp
linear.cpp
fused_linear_relu.cpp
graph.cpp
executor.cpp
model_json.cpp
optimizer.cpp
profiler.cpp
main.cpp
models/
mlp_demo.json
tests/
test_tensor.cpp
test_ops.cpp
test_graph.cpp
benchmarks/
bench_matmul.cpp
bench_mlp.cpp
third_party/
json/ (optional if vendoring)

````

---

## 5) Design Spec (Core Components)

### 5.1 Tensor
A Tensor should contain:
- `shape`: vector<int64_t> (e.g., [batch, features])
- `dtype`: float32 for v1
- `data`: contiguous buffer (std::vector<float> or aligned allocation)
- `strides`: optional; start with contiguous only

**Minimum operations**:
- `numel()`
- `reshape(new_shape)` (only if same numel)
- bounds checking in debug builds

**Key choices**:
- Start with float32 only (simple)
- Add float16/int8 later (quantization milestone)

---

### 5.2 Ops (Operators)
Each op has:
- input tensor ids (or pointers)
- output tensor id
- attributes (e.g., activation type, transpose flags)
- `run(context)` method

**Ops to implement** (in order):
1. `MatMul(A,B)->C` (2D)
2. `Add(A,bias)->C` (bias broadcast for 2D)
3. `ReLU(X)->Y`
4. `Softmax(X)->Y` (over last dimension)
5. `Linear(X,W,b)->Y` = MatMul + Add

---

### 5.3 Graph
Graph holds nodes and tensors:
- `std::vector<Node> nodes`
- `std::unordered_map<TensorId, Tensor>` (or tensor table)
- edges represented by node input/output ids

You will need:
- topological order (in many cases you’ll build it in order for v1)
- basic validation:
  - tensor existence
  - shape compatibility

---

### 5.4 Execution Context / Runtime
Executor responsibilities:
- allocate outputs/intermediates
- execute nodes sequentially
- (later) reuse buffers

Add simple profiler hooks:
- per-op timing
- total runtime
- memory usage estimates

---

### 5.5 Optimizer (Graph Passes)
Create a pass interface:
- `Pass::run(Graph&)`

Implement:
- **FuseLinearReLU**: replace `Linear -> ReLU` with `FusedLinearReLU`
- **MemoryReuse** (simplified):
  - track last use of each tensor id
  - free/reuse buffer after last use

---

## 6) Model Format (Simple JSON)

Start with a small JSON model definition. Example: 2-layer MLP.

`models/mlp_demo.json` (example schema)
```json
{
  "inputs": [
    { "name": "x", "shape": [1, 128] }
  ],
  "initializers": [
    { "name": "W1", "shape": [128, 256], "dtype": "f32", "data": "random_normal", "seed": 1 },
    { "name": "b1", "shape": [256],      "dtype": "f32", "data": "zeros" },
    { "name": "W2", "shape": [256, 10],  "dtype": "f32", "data": "random_normal", "seed": 2 },
    { "name": "b2", "shape": [10],       "dtype": "f32", "data": "zeros" }
  ],
  "nodes": [
    { "op": "Linear",  "inputs": ["x", "W1", "b1"], "outputs": ["h1"] },
    { "op": "ReLU",    "inputs": ["h1"],           "outputs": ["a1"] },
    { "op": "Linear",  "inputs": ["a1", "W2", "b2"], "outputs": ["logits"] },
    { "op": "Softmax", "inputs": ["logits"],       "outputs": ["y"] }
  ],
  "outputs": ["y"]
}
````

**Notes**

* For v1, you can generate initializer data in code (`random_normal`, `zeros`)
* Later you can support loading raw arrays from file

---

## 7) Build System (CMake)

### 7.1 CMakeLists (skeleton)

Create `CMakeLists.txt` like:

```cmake
cmake_minimum_required(VERSION 3.20)
project(mini_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MINII_ENABLE_TESTS "Build tests" ON)
option(MINII_ENABLE_OPENMP "Enable OpenMP" ON)

add_library(miniinfer
  src/tensor.cpp
  src/graph.cpp
  src/executor.cpp
  src/optimizer.cpp
  src/profiler.cpp
  src/model_json.cpp
  src/ops/matmul.cpp
  src/ops/add.cpp
  src/ops/relu.cpp
  src/ops/softmax.cpp
  src/ops/linear.cpp
  src/ops/fused_linear_relu.cpp
)

target_include_directories(miniinfer PUBLIC include)

if(MINII_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(miniinfer PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(miniinfer PUBLIC MINII_USE_OPENMP=1)
  endif()
endif()

add_executable(mini_infer_cli src/main.cpp)
target_link_libraries(mini_infer_cli PRIVATE miniinfer)

# Tests (optional)
if(MINII_ENABLE_TESTS)
  enable_testing()
  # You can vendor googletest or use FetchContent here
  # add_subdirectory(third_party/googletest)
  # add_executable(mini_tests tests/test_ops.cpp ...)
  # target_link_libraries(mini_tests PRIVATE miniinfer gtest_main)
  # add_test(NAME mini_tests COMMAND mini_tests)
endif()
```

---

## 8) VS Code Setup

### 8.1 Recommended `.vscode/settings.json`

```json
{
  "cmake.sourceDirectory": "${workspaceFolder}",
  "cmake.buildDirectory": "${workspaceFolder}/build",
  "cmake.configureOnOpen": true,
  "C_Cpp.default.cppStandard": "c++17"
}
```

### 8.2 Build + Run (CMake Tools)

1. Open folder in VS Code
2. `Ctrl/Cmd+Shift+P` → **CMake: Configure**
3. Select a kit (your compiler)
4. **CMake: Build**
5. Run:

   * `./build/mini_infer_cli --model models/mlp_demo.json --bench`

---

## 9) CLI (Command Line Interface) Spec

Your `main.cpp` should support:

* `--model path/to/model.json`
* `--input random` or `--input file`
* `--opt fuse` (enable fusion pass)
* `--opt memreuse` (enable memory reuse)
* `--bench` (run benchmark loop N times)
* `--print` (print output tensor summary)

Example usage:

```bash
./build/mini_infer_cli --model models/mlp_demo.json --opt fuse --bench
```

---

## 10) Correctness Strategy (How to Know It Works)

### 10.1 Reference checks

Implement a “reference mode”:

* naive matmul (triple loop)
* stable softmax
* compare outputs (max abs error)

### 10.2 Numeric tolerances

For float32:

* `max_abs_error < 1e-4` for small networks

### 10.3 Unit tests (minimum)

* Tensor reshape validity
* MatMul shape rules
* Softmax sums to 1 across last dim
* ReLU non-negativity

---

## 11) Performance Strategy (How to Make It Fast)

### 11.1 MatMul baseline

Implement:

* cache-friendly loops (i-j-k or i-k-j depending)
* optional OpenMP parallel over `i` (rows)

### 11.2 Avoid extra allocations

Common performance killer: allocating/freeing vectors in every op.
Fix:

* preallocate output buffers where possible
* reuse intermediate buffers (memory planning pass)

### 11.3 Operator fusion

Fusing `Linear + ReLU`:

* compute matmul + bias add and apply relu in one pass
* reduces memory writes/reads of intermediate

---

## 12) Benchmarking (What to Report)

### 12.1 MatMul benchmark

Measure:

* time (ms)
* GFLOPs (optional)
* speedup with OpenMP

### 12.2 End-to-end model benchmark

Run 100–1000 iterations, report:

* avg latency (ms)
* p50/p95 (optional)
* speedup after optimizations

**Store results** in `benchmarks/results.md`:

* CPU info
* compiler flags
* matrix sizes
* before/after tables

---

## 13) “Stretch Goals” (Optional, Big Resume Boost)

Pick ONE:

1. **ONNX import (partial)**

   * parse a tiny subset: Gemm/Relu/Softmax
2. **Quantization (INT8)**

   * per-tensor scale, int8 matmul accumulation to int32
3. **KV cache / attention (LLM-ish)**

   * implement scaled dot-product attention forward (small)
4. **Threadpool**

   * custom threadpool instead of OpenMP

---

## 14) Definition of Done (Shipping Checklist)

### Functionality

* [ ] Loads JSON model
* [ ] Runs inference end-to-end
* [ ] Produces stable outputs
* [ ] Has at least one demo model

### Engineering

* [ ] Unit tests runnable
* [ ] Benchmarks runnable
* [ ] Profiler output per op
* [ ] Clean build instructions

### Optimization

* [ ] Fusion pass implemented
* [ ] Memory reuse or multithreading implemented
* [ ] Documented speedup numbers

---

## 15) Suggested Resume Entry (Template)

**MiniInfer: C++ Neural Network Inference Engine**

* Built a C++17 inference runtime supporting tensors, computation graphs, and forward-pass execution for MLP models
* Implemented operator fusion (Linear+ReLU) and buffer reuse to reduce memory traffic and allocations
* Added benchmarking/profiling and achieved **X× latency reduction** on CPU compared to baseline execution

(When you add CUDA later)

* Accelerated MatMul/ReLU with CUDA kernels achieving **Y×** speedup vs CPU

---

## 16) Next Step

Create the repo structure and start Milestone A:

1. `tensor.h/.cpp`
2. `matmul.cpp` (naive + optimized)
3. `relu.cpp`
4. small hardcoded test in `main.cpp`
