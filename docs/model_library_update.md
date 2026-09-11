## Model and Library Update Notes

This project currently uses TensorFlow and Keras for production inference and training.

The environment files now target a more recent stack:

- Python 3.11
- TensorFlow 2.20
- Keras 3
- scikit-learn 1.7
- pandas 2.2
- scipy 1.15
- matplotlib 3.10
- firebase-admin 7.1

For IMU time series experiments, the codebase now exposes additional architecture choices:

- `inceptiontime` for jump type classification
- `tcn` for landing success prediction

The environment also includes `aeon` to prepare CPU-friendly experiments with non-neural time series classifiers such as MiniRocket and Hydra.

Recommended next experiment order:

1. Compare `transformer` vs `inceptiontime` on jump type.
2. Compare `lstm` vs `tcn` on success prediction.
3. Run the new `benchmark` command with `MiniRocket` on the same dataset split.
4. Compare `MiniRocket` and `Hydra` through the shared CLI benchmark entrypoint.
