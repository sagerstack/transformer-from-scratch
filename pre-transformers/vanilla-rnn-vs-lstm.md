# Vanilla RNN vs LSTM

## Disadvantages of RNN

### 1. Sequential Processing — Can't Parallelize

RNNs must process tokens one at a time because each step depends on the previous hidden state:

```
h1 = RNN("I", h0)        <- must finish before
h2 = RNN("love", h1)     <- this can start before
h3 = RNN("cats", h2)     <- this can start
```

You can't compute h3 without h2, and h2 without h1. For a 1000-word sequence, that's 1000 sequential operations. GPUs are built for parallel computation, but RNNs can't take advantage of that.

### 2. Long-Range Dependency Problem

Information from early tokens gets diluted as the sequence grows. By the time the RNN reaches word 500, the hidden state has been transformed so many times that information about word 1 is largely lost.

```
"The cat that sat on the mat that was near the door that ... ate the fish"
  ^                                                          ^
  word 1                                                    word 500
  (who ate the fish? the cat — but RNN has mostly forgotten it)
```

### Vanishing/Exploding Gradients — Same Problem, Two Views

The long-range dependency problem and vanishing/exploding gradients are the same problem viewed from two angles.

**Forward pass perspective -> long-range dependency problem**

The hidden state is multiplied by the weight matrix `W_h` at every step:

```
h1 = sigma(W_h . h0 + ...)
h2 = sigma(W_h . h1 + ...)
h3 = sigma(W_h . h2 + ...)
...
h100 = sigma(W_h . h99 + ...)
```

Information from h1 passes through 99 transformations to reach h100. Each multiplication and activation squashes or distorts the signal. By h100, the contribution of h1 is negligible.

**Backward pass perspective -> vanishing/exploding gradients**

During backpropagation, the gradient of the loss at step 100 with respect to h1 involves a chain of multiplied derivatives:

```
dL/dh1 = dL/dh100 . dh100/dh99 . dh99/dh98 . ... . dh2/dh1
```

Each `dh_t/dh_{t-1}` involves multiplying by `W_h`. After 99 multiplications:

- If the effective multiplier is < 1 at each step -> gradient **vanishes** -> early words don't get updated -> network can't learn long-range patterns
- If the effective multiplier is > 1 at each step -> gradient **explodes** -> training becomes unstable

**Same root cause:** repeated multiplication by `W_h` across many steps. The forward pass loses information, the backward pass loses gradients. Two symptoms of one disease.

---

## LSTM — Long Short-Term Memory

LSTMs add a **cell state** with additive updates (instead of multiplicative), which lets gradients flow more easily.

### Pros

- **Solves vanishing gradient** — the cell state acts as a highway where information flows through addition, not multiplication. Gradients can pass through unchanged via the forget gate
- **Selective memory** — three gates (forget, input, output) learn what to remember, what to add, and what to expose. The network decides per-step what's relevant
- **Better long-range than vanilla RNNs** — can practically handle dependencies over hundreds of steps (vs tens for basic RNNs)

### Cons

- **Still sequential** — same as RNN, must process one token at a time. Can't parallelize across sequence length. Training is slow on long sequences
- **Long-range has limits** — better than RNNs but still degrades over very long sequences (thousands of tokens). The cell state helps but isn't perfect
- **Complex and slow per step** — each step computes 3 gates + cell update = 4 matrix multiplications (vs 1 for vanilla RNN). More parameters, more computation per token
- **Fixed-size hidden state** — all information about the sequence must fit in one vector. No matter how complex the sequence, the bottleneck is the same size

### The Three Gates

An LSTM cell has three gates and a cell state. At each step, the input is the current word `x_t` and the previous hidden state `h_{t-1}`.

**Cell state `C` — the memory highway**

This is the key idea. It runs across all steps with only addition and element-wise multiplication — no `W_h` chain, which is why gradients survive.

```
C_{t-1} --[x forget]--[+ new info]----> C_t
```

**1. Forget gate — what to erase from memory**

```
f_t = sigma(W_f . [h_{t-1}, x_t])     -> values between 0 and 1
C_t = f_t * C_{t-1}                   -> 0 = forget, 1 = keep
```

Example: translating a sentence, you hit a period. The forget gate learns to clear the subject of the previous sentence since it's no longer relevant.

**2. Input gate — what new information to store**

```
i_t = sigma(W_i . [h_{t-1}, x_t])           -> which cells to update (0 to 1)
C_tilde = tanh(W_C . [h_{t-1}, x_t])        -> candidate new values (-1 to 1)
C_t = f_t * C_{t-1} + i_t * C_tilde         -> erase old + write new
```

Example: a new subject appears ("The dog"). The input gate learns to write "dog" into the cell positions that track the current subject.

**3. Output gate — what to expose from memory**

```
o_t = sigma(W_o . [h_{t-1}, x_t])     -> which parts of cell state to output
h_t = o_t * tanh(C_t)                 -> filtered version of memory
```

The cell state might store both subject and tense, but the next layer only needs the subject right now. The output gate filters what's relevant for the current prediction.

**Full picture for one step:**

```
         forget       input        output
         gate         gate         gate
          |            |            |
C_{t-1} -> [x f_t] -> [+ i_t*C_tilde] -> C_t
                                          |
                                    [tanh -> x o_t] -> h_t
```

### How Hidden State Flows Between Gates

The gates don't run sequentially with hidden state flowing between them. All three gates compute **in parallel** from the same two inputs: `h_{t-1}` and `x_t`.

```
                    h_{t-1}  +  x_t
                   /        |        \
                  v         v         v
            forget gate  input gate  output gate
            f_t = sigma  i_t = sigma  o_t = sigma
                  |         |         |
C_{t-1} --[x f_t]--[+ i_t * C_tilde]---> C_t
                                           |
                                     [tanh -> x o_t] -> h_t
                                                         |
                                              passed to next time step
```

The flow is:

1. `h_{t-1}` and `x_t` enter -> all three gates compute simultaneously
2. Forget gate and input gate update the **cell state**: `C_t = f_t * C_{t-1} + i_t * C_tilde`
3. Output gate filters the updated cell state to produce the **new hidden state**: `h_t = o_t * tanh(C_t)`
4. `h_t` and `C_t` both pass to the next time step

`h_{t-1}` is an **input** to all gates, not something that flows through them. The gates are independent functions that all read the same inputs and each contribute to a different part of the update.

### Cell State vs Hidden State

Both carry information, but they serve different roles:

**Cell state `C_t`** — the full, unfiltered long-term memory. It stores everything the LSTM has decided to remember. But it's **internal** — nothing outside the LSTM cell sees it directly.

**Hidden state `h_t`** — a filtered, compressed view of the cell state. This is what goes **out** to:
- The next time step (as input to the gates)
- The prediction layer (to generate output)
- The attention mechanism (in seq2seq models, the encoder hidden states that the decoder attends to are `h_t`, not `C_t`)

```
C_t = full memory (internal, private)
h_t = o_t * tanh(C_t) = filtered memory (external, public)
```

The cell state holds **more** information, but the hidden state is what the rest of the network works with. The output gate learns which parts of the full memory are relevant to expose at each step.

### Why LSTM h_t is More Informative Than Vanilla RNN h_t

Even though LSTM's `h_t` is filtered, it contains **more useful** information than vanilla RNN's `h_t`.

In a vanilla RNN, every step **overwrites** the hidden state through a full matrix multiplication. By step 100, information from step 1 has been multiplied by `W_h` 99 times. It's mostly destroyed.

In an LSTM, the cell state preserves information through **addition**. If the forget gate says `f_t = 1` (keep everything) and input gate says `i_t = 0` (add nothing), information passes through **perfectly unchanged**. The cell state can carry information from step 1 all the way to step 100 with zero degradation.

Then `h_t = o_t * tanh(C_t)` filters this well-preserved memory.

```
Vanilla RNN h_t:  degraded information about everything
LSTM C_t:         well-preserved information about everything
LSTM h_t:         well-preserved information, filtered to what's relevant now
```

Filtered-but-well-preserved beats unfiltered-but-degraded.

---

The fundamental issues of RNNs/LSTMs remain: sequential processing + information compressed into a fixed vector. Transformers eliminate both — parallel processing via self-attention, and direct access to all positions instead of squeezing everything through one vector.
