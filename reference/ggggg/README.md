# GGGGG reference model

This is the last small model used before switching to the full Hg-DNA
Hamiltonian. It has five G-C base pairs and includes HOMO and LUMO in the same
20-orbital matrix.

Run the three-bias calculation from the repo root:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m reference.ggggg.bias_voltage_comparison \
  --sequence GGGGG \
  --bias-voltages -0.5 0 0.5 \
  --gamma-left 1 \
  --gamma-right 1 \
  --d0 0.1 \
  --alpha 0.3 \
  --tolerance 0.1 \
  --max-iterations 250 \
  --energy-min -10 \
  --energy-max 10 \
  --energy-step 0.01 \
  --trace-energy -4.0 \
  --output-dir outputs/ggggg_bias_voltage_comparison
```

The three cases should converge in 85, 119, and 85 iterations.

Run the zero-bias calculation with:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m reference.ggggg.full_orbital_dos_weighted \
  --sequence GGGGG \
  --gamma-left 1 \
  --gamma-right 1 \
  --d0 0.1 \
  --alpha 0.3 \
  --tolerance 0.1 \
  --max-iterations 250 \
  --energy-min -10 \
  --energy-max 10 \
  --energy-step 0.01 \
  --trace-energy -4.0 \
  --output-dir outputs/sequence_ggggg_20_orbital_dos_weighted
```
