# Implementation Plan: IDPS-CRP

## Goal Description
Implement the **IDPS-CRP** architecture for Whole Slide Image classification in the `my-code01` directory. The structure will mirror the `ips` reference codebase (`architecture`, `data`, `training`, `config`) but adapted for the "Sparse Re-computation" method and CRP support.

## User Review Required
> [!IMPORTANT]
> This implementation will create a new directory structure in `my-code01`. It typically requires a dataset (WSI features or patches). I will mock the data loading or assume a standard format unless specified.

## Proposed Changes

### Structure Setup
Replicate the organized structure of `ips`:
#### [NEW] [architecture](file:///home/axel/Downloads/udsbooks/M2/Thesis/latex-sources/brainstorm/paper-code/Idps-crp/code01/my-code01/architecture)
- `idps_net.py`:
    - [ ] `scouting_pass()`: Pass 1 logic (NoGrad + Optional Downscale input).
    - [ ] `training_pass()`: Pass 2 logic (Grad + HighRes input).
    - [ ] `forward()`: Wrapper deciding flow.
- `transformer.py`: The attention mechanism.

#### [NEW] [training](file:///home/axel/Downloads/udsbooks/M2/Thesis/latex-sources/brainstorm/paper-code/Idps-crp/code01/my-code01/training)
- `trainer.py`: The training loop.
    - [ ] Implement `map_indices(idx_low, scale)` function.
    - [ ] Logic: `idx = model.scout(low_res_wsi); hi_res_patches = data.extract(idx); model.learn(hi_res_patches)`.

#### [NEW] [data](file:///home/axel/Downloads/udsbooks/M2/Thesis/latex-sources/brainstorm/paper-code/Idps-crp/code01/my-code01/data)
- `loader.py`:
    - [ ] `WSIWrapper` class: Methods `get_low_res()` and `get_patch_high_res(idx)`.



#### [NEW] [root](file:///home/axel/Downloads/udsbooks/M2/Thesis/latex-sources/brainstorm/paper-code/Idps-crp/code01/my-code01)
- `main.py`: Entry point.
- `requirements.txt`: Dependencies (including `zennit` for CRP).

## Verification Plan

### Automated Tests
I will create a dummy test script `test_model.py` to verify the gradient flow and shape correctness.
- **Command**: `python my-code01/test_model.py`
- **What it tests**:
    1.  Input shape handling (Bag of patches).
    2.  IPS Selection (Pass 1) correctness on *downsampled* input.
    3.  Re-computation (Pass 2) on *high-res* input.
    4.  Gradient flow through re-computed path.


### Manual Verification
- Run `python my-code01/main.py --dry-run` to ensure the pipeline runs for one epoch on random data.
