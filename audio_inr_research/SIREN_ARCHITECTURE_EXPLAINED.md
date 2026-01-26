# SIREN Architecture: Complete Explanation

## Table of Contents

1. [What is SIREN?](#what-is-siren)
2. [Architecture Diagram](#architecture-diagram)
3. [Key Components](#key-components)
4. [Code Walkthrough](#code-walkthrough)
5. [Training Process](#training-process)
6. [How It Works for Audio](#how-it-works-for-audio)

---

## What is SIREN?

**SIREN** stands for **Sine Representation Network**. It's an Implicit Neural Representation (INR) that uses **sine activation functions** instead of traditional ReLU/Tanh activations.

### Why Sine Activations?

- **Smooth derivatives**: Perfect for representing continuous signals (audio, images, 3D shapes)
- **High-frequency details**: Can capture fine details better than ReLU
- **Periodic properties**: Natural fit for audio signals

---

## Architecture Diagram

```
INPUT: Audio Coordinates (time points)
          ↓
    [B, N, 1]  (Batch, Sequence Length, 1D coordinate)
          ↓
┌─────────────────────────────────────────────────────┐
│                  SIREN NETWORK                      │
│                                                     │
│  ┌──────────────────────────────────────────┐     │
│  │  Layer 1: SineLayer (First Layer)        │     │
│  │  Input: 1 → Hidden: 256                  │     │
│  │  ω₀ = 30 (frequency scaling)             │     │
│  │                                           │     │
│  │  W₁: Linear(1 → 256)                     │     │
│  │  Output = sin(ω₀ × W₁(x))               │     │
│  │                                           │     │
│  │  Special Init: U(-1/n, 1/n)              │     │
│  └──────────────────────────────────────────┘     │
│          ↓                                          │
│    [B, N, 256]                                      │
│          ↓                                          │
│  ┌──────────────────────────────────────────┐     │
│  │  Layer 2: SineLayer (Hidden)             │     │
│  │  Input: 256 → Hidden: 256                │     │
│  │                                           │     │
│  │  W₂: Linear(256 → 256)                   │     │
│  │  Output = sin(ω₀ × W₂(x))               │     │
│  │                                           │     │
│  │  Init: U(-√6/n/ω₀, √6/n/ω₀)             │     │
│  └──────────────────────────────────────────┘     │
│          ↓                                          │
│    [B, N, 256]                                      │
│          ↓                                          │
│  ┌──────────────────────────────────────────┐     │
│  │  Layer 3: SineLayer (Hidden)             │     │
│  │  (Same as Layer 2)                       │     │
│  └──────────────────────────────────────────┘     │
│          ↓                                          │
│  ┌──────────────────────────────────────────┐     │
│  │  Layer 4: SineLayer (Hidden)             │     │
│  │  (Same as Layer 2)                       │     │
│  └──────────────────────────────────────────┘     │
│          ↓                                          │
│    [B, N, 256]                                      │
│          ↓                                          │
│  ┌──────────────────────────────────────────┐     │
│  │  Final Layer: Linear (No activation)     │     │
│  │  Input: 256 → Output: 1                  │     │
│  │                                           │     │
│  │  W_final: Linear(256 → 1)                │     │
│  │  Output = W_final(x)                     │     │
│  └──────────────────────────────────────────┘     │
│          ↓                                          │
└─────────────────────────────────────────────────────┘
          ↓
OUTPUT: Audio Amplitude Values
    [B, N, 1]  (Batch, Sequence Length, Amplitude)
```

---

## Key Components

### 1. **SineLayer** - The Building Block

```python
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0        # Frequency scaling factor
        self.is_first = is_first       # Different init for first layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: Uniform(-1/n, 1/n)
                self.linear.weight.uniform_(
                    -1 / self.linear.in_features,
                    1 / self.linear.in_features
                )
            else:
                # Hidden layers: Uniform(-√6/n/ω₀, √6/n/ω₀)
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, input):
        # The key: sin activation with frequency scaling
        return torch.sin(self.omega_0 * self.linear(input))
```

**What's happening?**

- `omega_0` (ω₀) controls the frequency of sine waves
- Higher ω₀ = captures finer details
- Special weight initialization ensures stable gradients
- First layer uses different init because it processes raw coordinates

### 2. **SIREN Network** - Stacking SineLayers

```python
class SIREN(BaseINR):
    def __init__(self, in_features=1, out_features=1,
                 hidden_features=256, num_layers=4,
                 first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        self.net = []

        # First layer (special)
        self.net.append(
            SineLayer(in_features, hidden_features,
                     is_first=True, omega_0=first_omega_0)
        )

        # Hidden layers (all the same)
        for _ in range(num_layers - 1):
            self.net.append(
                SineLayer(hidden_features, hidden_features,
                         is_first=False, omega_0=hidden_omega_0)
            )

        # Final layer (linear, no activation)
        self.net.append(nn.Linear(hidden_features, out_features))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)
```

---

## Code Walkthrough

### Example: Processing Audio with SIREN

Let's trace through a real example step by step:

```python
# Setup
import torch
from src.architectures.models import SIREN

# Create SIREN model
model = SIREN(
    in_features=1,          # 1D time coordinate
    out_features=1,         # 1D amplitude output
    hidden_features=256,    # 256 neurons per hidden layer
    num_layers=5,          # 5 sine layers
    first_omega_0=30.0,    # Frequency for first layer
    hidden_omega_0=30.0    # Frequency for hidden layers
)

# Create input: 16000 time coordinates (1 second at 16kHz)
batch_size = 1
seq_len = 16000
coords = torch.linspace(-1, 1, seq_len).reshape(batch_size, seq_len, 1)

print(f"Input shape: {coords.shape}")  # [1, 16000, 1]

# Forward pass
output = model(coords)
print(f"Output shape: {output.shape}")  # [1, 16000, 1]
```

### Step-by-Step Forward Pass

```python
# Let's manually trace what happens:

x = coords  # [1, 16000, 1]

# Layer 1: First SineLayer
# x → Linear(1, 256) → multiply by ω₀=30 → sin()
x = model.net[0](x)
print(f"After Layer 1: {x.shape}")  # [1, 16000, 256]

# Layer 2: SineLayer
x = model.net[1](x)
print(f"After Layer 2: {x.shape}")  # [1, 16000, 256]

# Layer 3: SineLayer
x = model.net[2](x)
print(f"After Layer 3: {x.shape}")  # [1, 16000, 256]

# Layer 4: SineLayer
x = model.net[3](x)
print(f"After Layer 4: {x.shape}")  # [1, 16000, 256]

# Layer 5: Final Linear (no activation)
x = model.net[4](x)
print(f"Final Output: {x.shape}")  # [1, 16000, 1]
```

---

## Training Process

### Overview Diagram

```
┌────────────────────────────────────────────────────┐
│                TRAINING PIPELINE                    │
└────────────────────────────────────────────────────┘

1. DATA PREPARATION
   ┌─────────────────────┐
   │  Load Audio File    │  →  [16000] samples (1 second)
   └─────────────────────┘
           ↓
   ┌─────────────────────┐
   │  Downsample by 4x   │  →  [4000] samples (low-res)
   └─────────────────────┘
           ↓
   ┌─────────────────────┐
   │ Generate Coords     │  →  [16000, 1] time points
   └─────────────────────┘

2. FORWARD PASS
   ┌─────────────────────┐
   │  SIREN(coords)      │  →  [16000, 1] predictions
   └─────────────────────┘

3. LOSS COMPUTATION
   ┌─────────────────────┐
   │  MSE(pred, gt)      │  →  scalar loss
   │  + STFT Loss        │
   └─────────────────────┘

4. BACKWARD PASS
   ┌─────────────────────┐
   │  loss.backward()    │  →  compute gradients
   └─────────────────────┘

5. OPTIMIZATION
   ┌─────────────────────┐
   │  optimizer.step()   │  →  update weights
   └─────────────────────┘

Repeat for all batches → Repeat for all epochs
```

### Training Code

```python
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # 1. Get data
        coord = batch['coord'].to(device)      # [B, 16000, 1]
        gt = batch['gt'].to(device)            # [B, 16000, 1]

        # 2. Forward pass: SIREN predicts amplitude at each coordinate
        optimizer.zero_grad()
        pred = model(coord)                     # [B, 16000, 1]

        # 3. Compute loss
        loss = loss_fn(pred, gt)                # scalar

        # 4. Backward pass
        loss.backward()                         # compute gradients

        # 5. Update weights
        optimizer.step()                        # apply gradients

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

### What Happens During Training?

```
Epoch 1:
  Batch 1:
    coords → SIREN → pred → loss(pred, gt) → gradients → update
    Loss: 0.1234

  Batch 2:
    coords → SIREN → pred → loss(pred, gt) → gradients → update
    Loss: 0.1156

  ...

  Average Loss: 0.1100

Epoch 2:
  (Repeat with updated weights)
  Average Loss: 0.0987

...

Epoch 100:
  Average Loss: 0.0123  ← Model has learned to represent audio!
```

---

## How It Works for Audio

### Conceptual Flow

```
PROBLEM: Represent continuous audio signal as a neural network

TRADITIONAL APPROACH:
  Audio = array of 16000 discrete samples
  ❌ Limited to fixed resolution
  ❌ Can't generate arbitrary resolutions

SIREN APPROACH:
  Audio = continuous function learned by neural network
  ✅ Query ANY time point
  ✅ Generate any resolution (super-resolution!)
  ✅ Smooth interpolation between points

┌────────────────────────────────────────────────┐
│          Traditional Audio Storage             │
│                                                │
│  Time:     0.0   0.0625  0.125   0.1875  ...  │
│  Sample:   0.1    0.4     0.2     -0.1   ...  │
│            [discrete array of 16000 values]    │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│           SIREN Representation                 │
│                                                │
│  Audio(t) = SIREN(t)                          │
│                                                │
│  Audio(0.0) = SIREN(0.0) = 0.1               │
│  Audio(0.0625) = SIREN(0.0625) = 0.4         │
│  Audio(0.0312) = SIREN(0.0312) = 0.25 ← NEW! │
│  Audio(any_t) = SIREN(any_t) ← Continuous!   │
└────────────────────────────────────────────────┘
```

### Real Example: Audio Super-Resolution

```python
# Training: Learn from low-res audio (4kHz)
lr_coords = torch.linspace(-1, 1, 4000).reshape(1, 4000, 1)  # 4000 points
lr_audio = load_downsampled_audio()  # [1, 4000, 1]

# Train SIREN to fit low-res audio
for epoch in range(100):
    pred = siren(lr_coords)
    loss = mse(pred, lr_audio)
    loss.backward()
    optimizer.step()

# Inference: Generate high-res audio (16kHz)
hr_coords = torch.linspace(-1, 1, 16000).reshape(1, 16000, 1)  # 16000 points
hr_audio = siren(hr_coords)  # [1, 16000, 1] ← Super-resolution!

# The network learned a continuous function,
# so we can query it at ANY resolution!
```

### Why SIREN Works Better Than MLP

```
MLP with ReLU:
  Input → ReLU → ReLU → ReLU → Output

  ✗ Piecewise linear (not smooth)
  ✗ Second derivative = 0 everywhere
  ✗ Poor at capturing high frequencies

  Output looks like: ___/‾‾‾\___/‾‾  (choppy)

SIREN with Sine:
  Input → sin(ω₀·x) → sin(ω₀·x) → sin(ω₀·x) → Output

  ✓ Infinitely smooth
  ✓ Non-zero derivatives at all orders
  ✓ Excellent at high frequencies (ω₀ controls this)

  Output looks like: ∿∿∿∿∿∿  (smooth)
```

---

## Key Hyperparameters

### 1. `omega_0` (ω₀) - Frequency Scaling

```python
# Low ω₀ = smooth, low-frequency signals
siren_smooth = SIREN(omega_0=1.0)   # Good for slow changes

# High ω₀ = detailed, high-frequency signals
siren_detailed = SIREN(omega_0=30.0)  # Good for audio (default)

# Very high ω₀ = can capture very fine details
siren_ultra = SIREN(omega_0=60.0)    # For very complex audio
```

### 2. `hidden_features` - Network Width

```python
# Small network = faster, less expressive
siren_small = SIREN(hidden_features=64)   # ~50K params

# Medium network = good balance (default)
siren_medium = SIREN(hidden_features=256)  # ~200K params

# Large network = more expressive, slower
siren_large = SIREN(hidden_features=512)   # ~800K params
```

### 3. `num_layers` - Network Depth

```python
# Shallow = fast, simple patterns
siren_shallow = SIREN(num_layers=3)  # Quick convergence

# Medium = good for most audio (default)
siren_medium = SIREN(num_layers=5)   # Balanced

# Deep = complex patterns, slower training
siren_deep = SIREN(num_layers=8)     # Very expressive
```

---

## Complete Training Example

```python
import torch
from src.architectures.models import SIREN
from src.loss_functions.losses import get_loss
import soundfile as sf

# 1. Load audio
audio, sr = sf.read('sample.wav')
audio = torch.tensor(audio).reshape(1, -1, 1)  # [1, N, 1]

# 2. Create coordinates
N = audio.shape[1]
coords = torch.linspace(-1, 1, N).reshape(1, N, 1)

# 3. Create model
model = SIREN(
    in_features=1,
    out_features=1,
    hidden_features=256,
    num_layers=5,
    first_omega_0=30.0,
    hidden_omega_0=30.0
)

# 4. Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

# 5. Training loop
model.train()
for epoch in range(1000):
    # Forward
    pred = model(coords)

    # Loss
    loss = loss_fn(pred, audio)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 6. Generate super-resolution audio (2x upsampling)
model.eval()
with torch.no_grad():
    hr_coords = torch.linspace(-1, 1, N*2).reshape(1, N*2, 1)
    hr_audio = model(hr_coords)

print(f"Original: {N} samples")
print(f"Super-res: {N*2} samples")
```

---

## Summary

### SIREN Architecture

- **Input**: Coordinates (time points) `[B, N, 1]`
- **Processing**: Stack of SineLayers with `sin(ω₀·x)` activation
- **Output**: Signal values (amplitudes) `[B, N, 1]`

### Key Advantages

1. **Continuous representation**: Query at any resolution
2. **Smooth gradients**: Perfect for optimization
3. **High-frequency details**: Better than ReLU networks
4. **Simple architecture**: Just linear layers + sine

### Training

1. Feed coordinates to SIREN
2. Compare predictions to ground truth
3. Backpropagate loss
4. Update weights
5. Repeat until convergence

The network learns a **continuous function** that represents the audio signal!
