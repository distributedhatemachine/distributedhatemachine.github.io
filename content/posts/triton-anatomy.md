---
title: "Anatomy of Triton"
author: "Egor"
date: "2025-11-24"
summary: "Dissecting Triton and writing small LLM purely in it"
description: "Dissecting Triton and writing small LLM purely in it"
toc: true
readTime: true
autonumber: true
math: true
tags: ["triton", "cuda"]
showTags: false
hideBackToTop: false
draft: false
---

# Disclaimer
In this post, I want to focus on Triton itself and won't go into the details of threads, blocks, grids, or warps. If you're not familiar with these concepts, I recommend reading [this post](https://modal.com/gpu-glossary/device-software/thread-block-grid)  first - and maybe checking out [this visualisation](https://gemini.google.com/share/c1fc651fa288) too.

# Fun part

## Why Triton?
If you've ever written GPU kernels in CUDA, you know the drill: think in threads, compute thread IDs, divide work manually, fight with memory coalescing, and hope the compiler doesn't betray you. This SIMT (Single Instruction, Multiple Threads) model is powerful, but also incredibly low-level.

Triton was created to change that. Instead of thinking in *threads*, Triton lets you think in *blocks* of data. It introduces a block-based programming paradigm where you write code that conceptually operates on a whole vector/tile at once, and the compiler takes care of generating efficient, coalesced, parallel GPU instructions.

In CUDA you think like this:

> "I am thread 5. I will load index 5."

In Triton, it becomes:

> "I am a block of pointers. I will load a block of data."

When you write something like:

```python
x = tl.load(ptr + offsets)
```


you're not loading a single element - you're loading an entire tensor tile directly into GPU registers. Triton's compiler then auto-vectorizes, auto-tiles, and arranges memory accesses to match modern GPU hardware. It's like magic and very powerful tool unless you need manual memory management or scheduling.

## What is kernel?

So what we are actually running on this threads/blocks? A kernel is simply a function that executes on the GPU - but unlike a normal function, it runs many times in parallel, once per thread or block (depending on the programming model). You write the logic for "one unit of work", and the GPU launches thousands of these units at once.

Every Triton operation needs TWO parts:

1. THE KERNEL (@triton.jit decorated function):
   - Runs ON the GPU
   - Defines the computation for ONE program instance
   - Has special restrictions (can't use regular Python features)
   - Uses tl.* functions (triton.language)
   
2. THE WRAPPER (regular Python function):
   - Runs ON the CPU
   - Sets up memory and launches the kernel
   - Defines the grid (how many parallel instances to run)
   - Passes parameters to the kernel

So very basic kernel could look like this:

```python
@triton.jit                   # 1. Decorator tells Triton to compile this
def my_kernel(
    input_ptr,                # 2. Pointers to GPU memory
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr  # 3. Compile-time constants (capitalized)
):
    pid = tl.program_id(0)    # 4. Which parallel instance am I?
    
    # 5. Calculate which elements THIS instance processes
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 6. Load data (with bounds checking - block size can be larger than the array)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # 7. Do computation
    y = x * 2
    
    # 8. Store results
    tl.store(output_ptr + offsets, y, mask=mask)
```

But let's ditch toy examples and write something more interesting. Like a toy LLM purely in Triton.

## LLM in Triton

Omiting embeddings, every Transformer is just bunch of Transformer Blocks. Each block has a attention layer (matmul + softmax), a feed-forward layer (matmul + gelu) and layer normalization.

### GELU

Let's start with easiest part - GELU.

GELU (Gaussian Error Linear Unit) is a non-linear activation function used in neural networks. It is defined as:

```python
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

```python
@triton.jit
def gelu_kernel(
    x_ptr, # Input pointer
    y_ptr, # Output pointer
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr # Elements per thread block
):
    # 1. Figure out which block/thread we are
    pid = tl.program_id(0) # ID of this parallel instance (0, 1, 2, ...)

    # 2. Calculate which elements THIS instance handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 3.Load data (with bounds checking - block size can be larger than the array)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # 4. Do computation
    y = 0.5 * x * (1.0 + tl.tanh(0.797885 * (x + 0.044715 * x * x * x)))
    
    # 5. Store results
    tl.store(y_ptr + offsets, y, mask=mask)

```

```python
def triton_gelu(x):
    y = torch.empty_like(x)
    n_elements = x.numel()

    # Choose block size (power of 2, typically 256-1024)
    BLOCK_SIZE = 1024

    #  Define grid: how many blocks do we need?
    # If n_elements=5000 and BLOCK_SIZE=1024, we need 5 blocks (5000/1024 = 4.88)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    gelu_kernel[grid](x, y, n_elements, BLOCK_SIZE)
    return y
```

### Softmax

```python
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Each program processes ONE row
    row_idx = tl.program_id(0)
    
    # Calculate starting position for this row
    row_start_ptr = input_ptr + row_idx * n_cols
    
    # Generate offsets for columns in this row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load the row (with masking for columns beyond n_cols)
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Step 1: Find max (for numerical stability)
    # Without this, exp(large_number) can overflow
    row_max = tl.max(row, axis=0)
    
    # Step 2: Subtract max and exponentiate
    row_shifted = row - row_max
    row_exp = tl.exp(row_shifted)
    
    # Step 3: Sum across the row
    row_sum = tl.sum(row_exp, axis=0)
    
    # Step 4: Normalize (divide by sum)
    row_softmax = row_exp / row_sum
    
    # Step 5: Store results
    output_row_start_ptr = output_ptr + row_idx * n_cols
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, row_softmax, mask=mask)
```
{{< details "Softmax wrapper" >}}

```python
def triton_softmax(x):
    n_rows, n_cols = x.shape
    
    # Block size must be >= n_cols, use next power of 2
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    y = torch.empty_like(x)
    
    # Launch one program per row
    grid = (n_rows,)
    softmax_kernel[grid](y, x, n_rows, n_cols, BLOCK_SIZE)
    
    return y
```
{{< /details >}}

### Matmul
This is hardest part so far because:
- we need to process 2D blocks
- we need to accumulate partial results
- we need [tiling](https://nichijou.co/cuda7-tiling/
) for efficiency

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,                     # Matrix dimensions: A is MxK, B is KxN, C is MxN
    stride_am, stride_ak,        # Strides for A
    stride_bk, stride_bn,        # Strides for B
    stride_cm, stride_cn,        # Strides for C
    BLOCK_M: tl.constexpr,       # Block size for M dimension
    BLOCK_N: tl.constexpr,       # Block size for N dimension
    BLOCK_K: tl.constexpr        # Block size for K dimension (reduction)
):
    # 2D grid: each program computes a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)  # Which row block
    pid_n = tl.program_id(1)  # Which column block
    
    # Generate offsets for the output block we're computing
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize pointers for A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension in tiles
    for k in range(0, K, BLOCK_K):
        # Load tiles from A and B
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        
        # Accumulate: C += A @ B for this tile
        acc += tl.dot(a, b)
        
        # Move to next tile in K dimension
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Convert accumulator to output type and store
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
```
{{< details "Matmul wrapper" >}}

```python
def triton_matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Define block sizes
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    
    # 2D grid: one program per output block
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N'])
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return c
```
{{< /details >}}


### Layernorm
{{< details "Hidden because it's very similar to softmax by nature" >}}

```python
@triton.jit
def layernorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Each program normalizes one row
    row_idx = tl.program_id(0)
    row_start = x_ptr + row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(row_start + col_offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n_cols
    
    # Compute variance
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    # Normalize
    x_norm = x_centered * rstd
    
    # Scale and shift
    weight = tl.load(weight_ptr + col_offsets, mask=mask)
    bias = tl.load(bias_ptr + col_offsets, mask=mask)
    y = x_norm * weight + bias
    
    # Store
    y_ptr_row = y_ptr + row_idx * n_cols
    tl.store(y_ptr_row + col_offsets, y, mask=mask)
```

```python
def triton_layernorm(x, weight, bias, eps=1e-5):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    layernorm_kernel[(n_rows,)](x, y, weight, bias, n_cols, eps, BLOCK_SIZE)
    return y
```
{{< /details >}}


### Finishing touches
{{< details "Now we can build whole LLM with those pieces." >}}

```python
class TritonAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.W_q = torch.randn(d_model, d_model, device='cuda', dtype=torch.float16) * 0.02
        self.W_k = torch.randn(d_model, d_model, device='cuda', dtype=torch.float16) * 0.02
        self.W_v = torch.randn(d_model, d_model, device='cuda', dtype=torch.float16) * 0.02
        self.W_o = torch.randn(d_model, d_model, device='cuda', dtype=torch.float16) * 0.02
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections using Triton
        Q = triton_matmul(x.view(-1, d_model), self.W_q).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = triton_matmul(x.view(-1, d_model), self.W_k).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = triton_matmul(x.view(-1, d_model), self.W_v).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Reshape for attention: (batch, n_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply softmax using Triton
        attn = triton_softmax(scores.view(-1, seq_len)).view(batch_size, self.n_heads, seq_len, seq_len)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        out = triton_matmul(out.view(-1, d_model), self.W_o).view(batch_size, seq_len, d_model)
        return out
```
```python
class TritonFFN:
    def __init__(self, d_model, d_ff):
        self.W1 = torch.randn(d_model, d_ff, device='cuda', dtype=torch.float16) * 0.02
        self.W2 = torch.randn(d_ff, d_model, device='cuda', dtype=torch.float16) * 0.02
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.view(-1, d_model)
        
        # First linear + GELU
        h = triton_matmul(x, self.W1)
        h = triton_gelu(h)
        
        # Second linear
        out = triton_matmul(h, self.W2)
        return out.view(batch_size, seq_len, -1)
```

```python
class TritonTransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.attention = TritonAttention(d_model, n_heads)
        self.ffn = TritonFFN(d_model, d_ff)
        
        self.ln1_weight = torch.ones(d_model, device='cuda', dtype=torch.float16)
        self.ln1_bias = torch.zeros(d_model, device='cuda', dtype=torch.float16)
        self.ln2_weight = torch.ones(d_model, device='cuda', dtype=torch.float16)
        self.ln2_bias = torch.zeros(d_model, device='cuda', dtype=torch.float16)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Self-attention with residual
        attn_out = self.attention.forward(x)
        x = x + attn_out
        x_flat = x.view(-1, d_model)
        x = triton_layernorm(x_flat, self.ln1_weight, self.ln1_bias).view(batch_size, seq_len, d_model)
        
        # FFN with residual
        ffn_out = self.ffn.forward(x)
        x = x + ffn_out
        x_flat = x.view(-1, d_model)
        x = triton_layernorm(x_flat, self.ln2_weight, self.ln2_bias).view(batch_size, seq_len, d_model)
        
        return x
```
```python
class TritonLLM:
    def __init__(self, vocab_size=1000, d_model=256, n_heads=8, n_layers=4, d_ff=1024, max_seq_len=128):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Token and position embeddings
        self.token_emb = torch.randn(vocab_size, d_model, device='cuda', dtype=torch.float16) * 0.02
        self.pos_emb = torch.randn(max_seq_len, d_model, device='cuda', dtype=torch.float16) * 0.02
        
        # Transformer blocks
        self.blocks = [TritonTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        
        # Output projection
        self.output_proj = torch.randn(d_model, vocab_size, device='cuda', dtype=torch.float16) * 0.02
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_emb[input_ids]
        pos_embeds = self.pos_emb[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = token_embeds + pos_embeds
        
        # Transformer blocks
        for block in self.blocks:
            x = block.forward(x)
        
        # Output projection
        logits = triton_matmul(x.view(-1, self.d_model), self.output_proj)
        logits = logits.view(batch_size, seq_len, self.vocab_size)
        
        return logits
```

{{< /details >}}

# Hard part

I hope you've enjoyed this and are curious to dive deeper into Triton - not just to write kernels, but to understand how it really works under the hood (especially since you could always just use [pre-written high-performance kernels]((https://github.com/zinccat/Awesome-Triton-Kernels))).

## Compilation pipeline (what happens when you @triton.jit a function)
- Triton parses the decorated Python function into a frontend AST and builds Triton-IR (a machine-independent IR tailored to Triton's block-level model).

- Triton lowers Triton-IR into GPU-specific IR (Triton-GPU / MLIR dialects). Optimizations and hardware-aware transforms happen here (tiling, shared-memory buffering, software pipelining, etc.).

- Lowering continues to LLVM IR, which is then used to generate PTX (NVIDIA) or other device code for the target backend (HIP/ROCm, MPS/Metal...).

- PTX (or the device binary) is JIT/linked into a device binary (CUBIN) that the runtime can launch. Triton caches compiled variants (specialization).  
*Because of constexprs and meta parameters, one Python kernel can produce many specialized binaries.*

More [here](https://pytorch.org/blog/triton-kernel-compilation-stages/) and [there](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/).

## Execution model - kernels, grids, and how they map to GPU hardware

- Triton exposes the kernel at the block/program level: every program instance (the work unit you index with tl.program_id) is analogous to a CUDA thread block / CTA but with Triton's own abstraction. Internally a program instance is executed cooperatively by a configurable number of threads (warps).

- `num_warps` controls how many warps (32-thread groups) are assigned per program instance. e.g. `num_warps`=8 => each program runs on 8 * 32 = 256 threads. `num_stages` tells the compiler how many pipeline stages to insert for software-pipelining (helpful for tiled GEMMs). 


- The grid you pass to kernel[grid] determines how many program instances to launch along each program axis; tl.program_id / tl.num_programs give the per-instance index/number.

Why **this** is magic I've talked before: Triton's block-level model frees you from thinking about individual threads - you reason in blocks/tiles and let the compiler parallelize across threads/warps inside each block. This is what makes Triton kernels compact to write while still being efficient.

## Memory model & synchronization

- Triton kernels use the usual GPU memory hierarchy: **global (HBM)**, **shared memory (SMEM)** (software-managed), and **registers**. tl.load/tl.store generate memory accesses; the compiler decides placement/tiling. Use masks to make loads/stores predicated (bounds-safe) — these become predicate instructions or masked loads/stores in generated PTX. 

- Shared memory / barriers: Triton exposes barrier primitives (e.g. `triton.language.debug_barrier`) and the compiler lowers higher-level sync to device barrier ops; some backends also provide lower-level mbarrier ops for fine control. Full global synchronization/block-to-block synchronization is limited - use atomics or communication primitives designed for inter-CTA communication if you need cross-block coordination (and beware of portability/ordering caveats). 

- Predicated/masked IO: masks avoid OOB accesses and are emitted as predicated accesses - cheaper than explicit branches in many cases.

## Key compiler optimizations knobs you should know

- `tl.constexpr` (compile-time constants): every constexpr in the signature produces a specialization. Use for block/tile sizes and loop bounds to let the compiler generate tighter code.

- `num_warps` (threads-per-program) and `num_stages` (software pipelining) - important for occupancy/latency hiding on newer SM architectures; increasing num_warps increases intra-program parallelism while `num_stages` pipelines loads with compute for tiled reductions/GEMMs. 

- Autotuning: `@triton.autotune` + `triton.Config` lets you supply multiple tile/warps/stage configs and benchmarks them to pick the best for the current shape/device. This is usually easier than hand-tuning every kernel. 

- Tensor ops & `tl.dot`/`tl.matmu`l: these intrinsics allow the compiler to target Tensor Cores / MMA instructions when available —-use them for inner products and small GEMM tiles to get huge speedups.

- Software pipelining / multi-buffering: the compiler can overlap loads/stores with compute across stages; `num_stages` selects the pipeline depth. Useful in matmul/attention kernels.

# tl.debug_barrier
Let's stop here and synchronize. There's plenty more to explore in Triton's optimization toolbox (just tell me if you want Part 2). I'm also cooking up a matching breakdown for Cute/TileLang, so keep an eye out!
