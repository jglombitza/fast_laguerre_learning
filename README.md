# Laguerre-Gaussian Mode Learning

This project implements deep learning models to decompose optical phase patterns into Laguerre-Gaussian (LG) mode coefficients using JAX and Keras.

## Overview

Laguerre-Gaussian modes are solutions to the paraxial wave equation with helical phase fronts, commonly used in optical physics (e.g., optical vortices, orbital angular momentum). This code:

1. **Generates synthetic datasets** of phase patterns from random LG mode superpositions
2. **Trains CNN models** to predict LG mode coefficients from phase images
3. **Implements custom JAX layers** for computing LG modes and reconstructing phase patterns
4. **Compares two architectures**:
   - Direct image-to-image reconstruction with embedded LG synthesis
   - Coefficient prediction with external reconstruction

## Key Features

- Custom `LGPhaseLayer`: Differentiable LG mode synthesis using JAX
- `JAXL2Norm`: L2 normalization layer for coefficient constraints
- Generalized Laguerre polynomial computation using explicit formulas
- Order 5 modes (15 total modes with indices satisfying 2p + |l| ≤ 4)


### Installation from conda env

If you prefer manual setup:

```bash
conda env create -f environment.yaml
conda activate laguerre_learning
```

### Verify GPU Support

After installation, verify CUDA/GPU availability:

```python
import jax
print(jax.devices())  # Should show GPU devices
```

## Usage

Run the main script:

```bash
python laguerre_poly.py
```

This will:
1. Generate 100k training and 10k validation samples
2. Display example phase patterns with mode compositions
3. Train two models (with early stopping via Ctrl+C supported)
4. Save reconstruction examples as PNG files

## Requirements

- **Python**: 3.12+
- **JAX**: 0.4.23 with CUDA 12.2 support
- **Keras**: 3.0+ (configured to use JAX backend)
- **NumPy, SciPy, Matplotlib**: For numerical computing and visualization
- **CUDA Toolkit**: 12.2
- **cuDNN**: 8.9

## Output Files

The script generates several output files:
- `pred_example_{0-3}.png`: Phase reconstruction from first model
- `rec_coeff_example_{0-3}.png`: Coefficient-based reconstruction

## Model Architectures

### Phase Model (Image-to-Image)
- Encoder: 3-block CNN with average pooling
- Decoder: Dense layers → LGPhaseLayer (custom synthesis)
- Loss: MSE on reconstructed phase + coefficient regularization

### Mode Model (Pure Coefficient Prediction)
- Deeper CNN encoder (4 blocks)
- Direct coefficient prediction with L2 normalization
- Loss: MSE on predicted coefficients

## Notes

- Training uses 64-sample batches
- Default: 50 epochs (interruptible)
- Models use ELU activations throughout
- Coefficient normalization ensures unit total power (∑|c|² = 1)
