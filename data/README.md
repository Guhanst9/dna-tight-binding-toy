# Data files

`xx7tg6.pdb` contains the atom order and partition numbers for the 7-base-pair
Hg-DNA model.

`xx7tg6.mat` contains the Hamiltonian. It is about 191 MB, so it is ignored by
Git and needs to be placed in this folder manually. The file should contain a
MATLAB variable named `xx7tg6` with shape `5083 x 5083`. Its values are in eV.

`DNATransmission_Decoherence.m` is the MATLAB transport code that came with
the model. It reads another file named `Parameters.txt`, which was not included
with the data.
