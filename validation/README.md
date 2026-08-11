# Lysozyme validation

`scripts/validate_profile.py` performs two checks on a completed profiling run:

1. It verifies successful AlphaFold completion, GPU samples, timing data, logs,
   a ranked model, and confidence output.
2. When given an experimental structure, it uses US-align to calculate C-alpha
   RMSD and TM-score for the predicted structure.

Results are written to `validation.json` inside the profile directory. No
accuracy threshold is imposed; the file preserves metrics for later analysis.

## Install US-align

[US-align](https://zhanggroup.org/US-align/) supports protein structure
alignment in PDB and mmCIF formats. Its
[help page](https://zhanggroup.org/US-align/help) documents all options and
metric definitions.

Download and compile the official single-file source:

```bash
curl -fsSL https://zhanggroup.org/US-align/bin/module/USalign.cpp -o /tmp/USalign.cpp && g++ -O3 -ffast-math -o /tmp/USalign /tmp/USalign.cpp
```

The resulting `/tmp/USalign` executable is temporary local tooling.

## Reference structure

The benchmark input uses the mature 129-residue hen egg-white lysozyme sequence
from [PDB 1LYZ](https://www.rcsb.org/structure/1LYZ), corresponding to
[UniProt P00698](https://www.uniprot.org/uniprotkb/P00698) residues 19-147.

Download the experimental mmCIF structure from RCSB:

```bash
curl -fsSL https://files.rcsb.org/download/1LYZ.cif -o /tmp/1LYZ.cif
```

The downloaded reference is temporary local validation data. Its SHA-256 hash
is recorded in `validation.json`.

## Run validation

Check profiling output without structural comparison:

```bash
python3 scripts/validate_profile.py profiles/RUN
```

Include comparison with the experimental structure:

```bash
python3 scripts/validate_profile.py profiles/RUN --reference /tmp/1LYZ.cif --usalign /tmp/USalign
```

Replace `profiles/RUN` with the generated profile directory. The validator uses
the ranked model in its top-level AlphaFold output directory rather than an
individual seed/sample model.

`validation.json` includes AF3 confidence output, number of GPU samples,
prediction and reference hashes, aligned residue count, RMSD, and both
length-normalised TM-scores. For this invocation, `tm_score_reference` is the
score normalised by the experimental structure length. US-align describes a
protein TM-score of at least 0.5 as indicating the same global fold; RMSD and
TM-score should still be interpreted alongside aligned length and AF3 confidence.
