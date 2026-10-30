# ERSRR MARS v3 validation result

Validation-selected result on the frozen internal groups; strict test remains untouched.

- Model: `ersrr_mars_full_unet_proposal_v3` / 14,268,915 parameters
- Samples: 768 train / 384 validation
- Best epoch: 5 / 5
- Validation recall at FPR <= 0.05: 0.062 (FPR 0.039)
- Validation AUROC / AP: 0.492 / 0.337
- Validation mask Dice: 0.290
- Checkpoint SHA-256: `a54529aa5ad45d1b8a15061a69645e7566a2d7b11eed9d2cf337b6e0dd4c022a`

## Decision

Freeze this validation-selected checkpoint before any strict-spatial evaluation.
