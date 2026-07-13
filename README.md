# how to run

Install dependencies:

```bash
python3 -m pip install numpy matplotlib pytest
```

Run one energy point:

```bash
python3 hamiltonian.py --pair gc --band homo --base-pairs 10 --gamma-left 0.5 --gamma-right 0.5 --energy -4.3 --d0 0.01 --tolerance 0.1 --max-iterations 100 --alpha 0.5 --show-basis --show-contacts --show-ldos --show-transport
```

Generate diagnostics:

```bash
python3 report.py --pair gc --band homo --base-pairs 10 --gamma-left 0.5 --gamma-right 0.5 --d0 0.01 --tolerance 0.1 --max-iterations 100 --alpha 0.5 --energy-min -7 --energy-max 2 --energy-points 1000 --trace-energy -4.3 --output-dir outputs/gc_homo
```

Run tests:

```bash
python3 -m pytest
```

## inputs

`--pair`

Base pair type used in the model Hamiltonian. Choices are `gc` or `at`.

`--band`

Which single-state model to use. Choices are `homo` or `lumo`.

`--base-pairs`

Number of DNA base pairs in the Hamiltonian. Default is `10`.

`--gamma-left`

Left contact strength in eV. Default is `0.5`.

`--gamma-right`

Right contact strength in eV. Default is `0.5`.

`--energy`

Energy point being evaluated in eV. This is required for `hamiltonian.py`.

`--d0`

Decoherence strength parameter in eV^2. Default is `0.01`.

`--tolerance`

Convergence tolerance in percent for both DOS and effective transmission. Default is `0.1`.

`--max-iterations`

Maximum number of self-consistent solver iterations, used to stop infinite loops if the solver does not converge. Default is `100`.

`--alpha`

Linear mixing parameter from the paper. It controls how much the newest decoherence self-energy update is trusted compared to the previous update. For example, `0.5` means 50% newest update and 50% previous update. It is unitless and must be between `0` and `1`. Default is `0.5`.

`--energy-min`

Minimum energy in eV for the report energy sweep.

`--energy-max`

Maximum energy in eV for the report energy sweep.

`--energy-points`

Number of energy points in the report sweep.

`--trace-energy`

Single energy point in eV used for the probe Gamma_B iteration and residue plots.

`--output-dir`

Folder where report CSV and PNG files are written.

`--show-basis`

Print the orbital basis labels.

`--show-contacts`

Print left contact, right contact, probe indices, and contact self-energy matrices.

`--show-ldos`

Print final LDOS values.

`--show-transport`

Print active probe matrix entries and probe broadening values.

## report outputs

`eigenvalues.csv`

Hamiltonian eigenvalues in eV.

`transmission_vs_energy.csv`

Coherent `T_LR` and decoherent `T_eff` for each energy point. Transmission is unitless.

`gamma_by_iteration.csv`

Probe `Gamma_B` values at each self-consistent iteration. `Gamma_B` is in eV.

`gamma_by_residue.csv`

Final converged probe `Gamma_B` for each internal residue/orbital. `Gamma_B` is in eV.

`transmission_coherent.png`

Coherent transmission vs energy with log y-axis.

`transmission_decoherent.png`

Decoherent transmission vs energy with log y-axis.

`gamma_by_iteration.png`

Probe `Gamma_B` vs self-consistent iteration with symlog y-axis so zero and small values are both visible.

`gamma_by_residue.png`

Final converged `Gamma_B` by residue with log y-axis.
