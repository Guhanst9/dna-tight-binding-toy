# how to run

```bash
python3 -m pip install numpy
python3 hamiltonian.py --pair gc --band homo --base-pairs 10 --gamma-left 0.5 --gamma-right 0.5 --energy -4.3 --d0 0.01 --tolerance 0.1 --max-iterations 100 --alpha 0.5 --show-basis --show-contacts --show-ldos --show-transport
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

Energy point being evaluated in eV. This is required.

`--d0`

Decoherence strength parameter in eV^2. Default is `0.01`.

`--tolerance`

Convergence tolerance in percent for both DOS and effective transmission. Always use default which is `0.1`.

`--max-iterations`

Maximum number of self-consistent solver iterations, used to stop infinite loops if the solver does not converge. Default is `100`.

`--alpha`

Linear mixing parameter from the paper. It controls how much the newest decoherence self-energy update is trusted compared to the previous update. For example, `0.5` means 50% newest update and 50% previous update. It is unitless and must be between `0` and `1`. Default is `0.5`.

`--show-basis`

Print the orbital basis labels.

`--show-contacts`

Print left contact, right contact, probe indices, and contact self-energy matrices.

`--show-ldos`

Print final LDOS values.

`--show-transport`

Print active probe matrix entries and probe broadening values.
