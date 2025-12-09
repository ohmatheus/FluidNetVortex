# **2D Fluid Simulation Approximation with CNN + C++ Runtime**

Complete roadmap, architecture, workflow, validation metrics, and integration strategy for a project that approximates a 2D fluid simulation using a convolutional neural network (CNN) trained on Blender Mantaflow data, then executed in real time inside a C++ engine using ONNX Runtime.

---

# **1. Project overview**

The objective is to create a real-time 2D fluid simulation approximator:

* **Training:** PyTorch (CNN/U-Net) on dataset exported from Blender/Mantaflow.
* **Inference:** ONNX Runtime in a C++ engine.
* **Rendering:** Simple 2D shading based on density.
* **Interaction:** User injects forces and density via mouse input.

**Grid resolution:** 128×128 (or 256×256).

**Model inputs:**

* Density(t)
* Velocity_X(t)
* Velocity_Y(t)
* Density(t-1)
* Source(t)                 #user input
* force_x(t), force_y(t)    #user input

**Model outputs:**

* Density(t+1)
* Velocity_X(t+1)
* Velocity_Y(t+1)

**Constraints:** real-time inference, clean architecture, demonstration-ready visuals.

---

# **2. Roadmap (First iterative development plan)**

### **Step 1 — Dataset generation (Blender/Mantaflow)**

* Export 1000+ short sequences of 20–50 frames.
* Save density and velocity fields per frame in `.npz`.

### **Step 2 — Python data pipeline**

* Create a custom PyTorch `Dataset` loader.
* Normalize density/velocity.
* Apply augmentations: rotations, flips, small noise.

### **Step 3 — CNN architecture**

* **small U-Net** (1–3M params).
* 7 input channels (3 for user inputs) → 3 output channels.

### **Step 4 — Training**

* Train with L2 loss + divergence penalty.
* Add gradient loss for sharp density transitions.
* Perform single-step + multi-step rollout evaluation.

### **Step 5 — Export to ONNX**

* Export with static shapes.
* Validate ONNX inference in Python.

### **Step 6 — C++ engine integration**

* ONNX Runtime (CPU/GPU).
* GLFW + OpenGL for rendering.
* Dear ImGui for debugging.
* Threaded inference system.

### **Step 7 — Rendering implementation**

* Upload density to a float texture.
* Colormap in shader.

### **Step 8 — User controls**

* Inject velocity with left drag.
* Inject density with right click.

---

# **3. Machine Learning architecture**

## **3.1 Model input/output shapes**

* Input tensor: `(B, 7, 128, 128)`
* Output tensor: `(B, 3, 128, 128)`

## **3.2 Architecture: Compact U-Net**

* Encoder: strided convolutions for downsampling.
* Bottleneck: residual blocks.
* Decoder: upsampling + convolution.
* Skip connections preserve fine details.

### **Activation & Norm**

* Use **GELU** or **LeakyReLU**.
* Use **InstanceNorm** or **LayerNorm**.

---

# **4. Loss functions**

Define the total loss:

```text
L = L2_density + α * L2_velocity + β * divergence_penalty + γ * gradient_loss
```

Where:

* **Density L2**: direct prediction accuracy.
* **Velocity L2**: correctness of vector field.
* **Divergence penalty**: encourages incompressibility.
* **Gradient loss**: preserves edges in smoke/density.

Typical coefficients:

* α = 1.0
* β = 0.1
* γ = 0.1

---

# **5. Dataset pipeline (Blender → PyTorch)**

## **5.1 Blender export strategy**

Store for each frame:

```
density_t.npy
velx_t.npy
vely_t.npy
```

Pack sequences into:

```
seq_000.npz
seq_001.npz
...
```

## **5.2 PyTorch dataset**

* Loads sequences.
* Builds training samples of: input `(d_t, vx_t, vy_t, d_t-1)` and target `(d_t+1, vx_t+1, vy_t+1)`.
* Supports random shifts, flips, and rotations.

---

# **6. Validation metrics**

Validation must go beyond single-frame accuracy.

### **6.1 Standard metrics**

* **MSE / MAE** on density and velocity.
* **PSNR / SSIM** on density.

### **6.2 Physics metrics**

* **Divergence norm**: `||div(vel_pred)||`.
* **Energy drift**: compare kinetic energy.

### **6.3 Rollout stability** (important!)

Simulate forward for 50–200 steps using only the model’s outputs.
Compute:

```
RMSE_rollout(T) = sqrt( sum_t ||pred_t - gt_t||^2 / T )
```

---

# **7. ONNX Export & validation**

## **7.1 Export example**

```python
torch.onnx.export(
    model.eval(),
    torch.randn(1,4,128,128),
    "fluid_model.onnx",
    opset_version=14,
    input_names=["input"],
    output_names=["density","velx","vely"],
)
```

## **7.2 Validate ONNX inference**

* Load ONNX model in Python via `onnxruntime`.
* Compare outputs vs PyTorch.
* Ensure numerical consistency.

---

# **8. C++ Integration architecture**

## **8.1 Technologies**

| Purpose           | Library                             |
| ----------------- | ----------------------------------- |
| Windowing + Input | GLFW (recommended) / SDL2 / SFML    |
| Rendering         | OpenGL (simple & portable)          |
| GUI/Debug         | Dear ImGui                          |
| Model Inference   | ONNX Runtime (CPU or CUDA provider) |

---

# **9. C++ Runtime loop**

### **State buffers**

* `density_curr`
* `velx_curr`
* `vely_curr`
* `density_prev`

### **Frame loop**

1. **Process input** (mouse → forces).
2. **Prepare input tensor** for ONNX.
3. **Run inference**.
4. **Swap buffers**.
5. **Upload density texture** to GPU.
6. **Render quad**.
7. **Draw UI** (debug info).

### **Threading**

Use a worker thread for model inference:

* Main thread: rendering + input.
* Worker: runs ONNX inference and updates next-state buffers.

---

# **10. Rendering implementation**

### **Texture setup**

* Create OpenGL texture with `GL_R32F`.
* Upload 128×128 (or 256×256)density field each frame.

### **Fragment shader (colormap)**

Maps density to color:

```glsl
float d = texture(densityTex, uv).r;
vec3 color = vec3(d * 0.1, d * 0.4, d * 1.0);
```

### **Filtering**

Two modes:

* `GL_NEAREST`: pixel-art look (grid visible).
* `GL_LINEAR`: smoother.

---

# **11. Example python model (simplified small U-Net)**

```python
import torch
import torch.nn as nn

class SmallUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(4, base, 3, padding=1), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2, 3, stride=2, padding=1), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv2d(base*2, base*4, 3, stride=2, padding=1), nn.GELU())
        self.bottleneck = nn.Sequential(nn.Conv2d(base*4, base*4, 3, padding=1), nn.GELU())
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base*8, base*2, 3, padding=1), nn.GELU()
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base*4, base, 3, padding=1), nn.GELU()
        )
        self.final = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        u2 = self.up2(torch.cat([b, e3], dim=1))
        u1 = self.up1(torch.cat([u2, e2], dim=1))
        out = self.final(torch.cat([u1, e1], dim=1))
        return out
```

---

# **12. Example ONNX export**

```python
model.eval()
dummy = torch.randn(1, 4, 128, 128)
torch.onnx.export(
    model,
    dummy,
    "fluid_sim.onnx",
    opset_version=14,
    input_names=["input"],
    output_names=["density","velx","vely"],
)
```

---

# **13. High-Level C++ skeleton**

```cpp
// Pseudocode (high-level)

Initialize GLFW + OpenGL
Initialize ImGui
Initialize ONNX Runtime session

Allocate state buffers: density, velx, vely, prev_density

while (!windowShouldClose) {
    processInput();

    // Prepare ONNX input
    float input[4][128][128] = ...;

    // Run inference
    session.Run(...);

    // Swap buffers
    density = density_next;

    // Upload texture
    glTexSubImage2D(...);

    // Render quad
    drawFullscreenQuad();

    // ImGui debug
    drawUI();

    glfwSwapBuffers();
}
```

---

# **14. User interaction design**

* **Left Click + Drag**: inject velocity.
* **Right Click**: inject density.
* **Sliders**:

  * Force strength
  * Brush radius
  * Viscosity scaling

---

# **15. Performance considerations**

* Keep model < 3M parameters for CPU RT.
* If GPU available: build ONNX Runtime with CUDA provider.
* Use quantization (INT8) if CPU struggles.
* Minimize memory copies between CPU/GPU.

---

# **16. Key risks & mitigation**

| Risk                 | Solution                                 |
| -------------------- | ---------------------------------------- |
| Rollout divergence   | Use rollout loss + curriculum learning   |
| CPU latency too high | Quantization or smaller U-Net            |
| ONNX unsupported ops | Stick to basic Conv/GELU/Upsample        |
| Dataset mismatch     | Normalize consistently across Python/C++ |

---

# **17. First checklist**

* [ ] Export 200+ sequences from Blender. (strat with 10)
* [ ] Implement PyTorch dataset loader.
* [ ] Implement and train small U-Net.
* [ ] Add divergence + gradient loss.
* [ ] Export ONNX and validate.
* [ ] Build C++ app with GLFW + OpenGL + ONNX Runtime.
* [ ] Implement force injection.
* [ ] Polish UI and rendering.

---


## Simulation choice and setup

### Primary simulation: Vortex field interaction

* High visual impact with swirling, coherent structures.
* Easy to simulate and approximate with CNNs.
* Should work perfectly on a 128×128/256×256 2D grid.
* Great for interactive demos: mouse drag = force injection.

**Blender setup parameters:**

* Domain: 128×128×1 or thin slice.
* Type: Gas (smoke) for smooth density advection.
* Vorticity: 0.4 (tune 0.2–1.0).
* Dissipation: ~0.99.
* Noise/Turbulence: 0.02–0.1.
* Emitters: 1–2 small circular density sources.
* Add a circular force field or animated inflow to drive rotation.
* Optionally include a small obstacle to create secondary vortices.
* Frame count per sequence: ~60.
* Generate 500–2000 sequences.
* Randomize emitter position, radius, strength, vortex center, turbulence.

**Export per frame:**

* density(t)
* vel_x(t), vel_y(t)
* density(t-1)
* source(t)
* force_x(t), force_y(t)

---

### Recommanded secondary simulation (optional later): Breaking Jet with Collider

* Impressive visually (water jet hitting obstacle).
* Harder for CNN due to high-frequency details.
* Keep as optional second demo.

---

## User interaction & runtime mapping

### Interaction fields

* **force_x / force_y:** per-cell velocity injection from mouse drag.
* **source:** density injection from mouse click.

Use Gaussian falloff around cursor. Keep brush radius between 4–16 px.

### Runtime pipeline

1. Apply small immediate modification to velocity/density for responsiveness.
2. Construct `source`, `force_x`, `force_y` grids.
3. Pack model input: *(density, vel_x, vel_y, density_prev, source, fx, fy)*.
4. Run ONNX inference.
5. Swap buffers and render.

---

## Training setup

### Model

* U‑Net, base channels 32–48 (~1–3M params).
* Inputs: 7 channels.
* Outputs: density(t+1), vel_x(t+1), vel_y(t+1).

### Losses

* L2 for density and velocity.
* Divergence penalty.
* Gradient loss for smoothness.
* Rollout loss (10–40 step closed-loop).

### Dataset

* 500–2000 sequences, 60 frames each.
* Batch size: 8–16.
* LR: 1e‑3 (AdamW).
* Use AMP.

### Validation Metrics

* One-step MSE.
* Rollout RMSE at 50 steps.
* Divergence norm.
* Long-term stability / energy drift.

---

## Rendering

* Density → color map (blue→cyan→white).
* Optional bloom/glow for high-density cores.
* Optional debug overlays: velocity field, vorticity map.
* Choose crisp (nearest) or smooth interpolation.

---

## References

* [https://arxiv.org/pdf/2412.10748](https://arxiv.org/pdf/2412.10748)
* [https://hhuiwangg.github.io/projects/awesome-neural-physics/](https://hhuiwangg.github.io/projects/awesome-neural-physics/)
* [https://arxiv.org/pdf/2006.08762](https://arxiv.org/pdf/2006.08762)
* [https://arxiv.org/pdf/1806.02071](https://arxiv.org/pdf/1806.02071)
* [https://proceedings.mlr.press/v70/tompson17a/tompson17a.pdf](https://proceedings.mlr.press/v70/tompson17a/tompson17a.pdf)
