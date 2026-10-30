# ERSRR MARS v3 validation result

Validation-selected result on the frozen internal groups; strict test remains untouched.

- Model: `ersrr_mars_full_unet_proposal_v3` / 14,268,915 parameters
- Samples: 768 train / 384 validation
- Best epoch: 7 / 14
- Validation recall at FPR <= 0.05: 0.180 (FPR 0.047)
- Validation AUROC / AP: 0.673 / 0.489
- Validation mask Dice: 0.307
- Checkpoint SHA-256: `95dbb87d9dfea752849bc0ec7fb6af9fe8743ad4c24fc0f1b6a5ca14f484f9a2`

## Decision

Freeze this validation-selected checkpoint before any strict-spatial evaluation.
