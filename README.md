# combination-kinetics-public

This is the basic simulation code for [REDACTED].

__environment.yml__  
This file specifies a conda environment that should include what is needed to run the code.
Use it with (for instance)
```bash
conda env create -n combination-kinetics -f environment.yml
```

__Snakefile__  
This is a makefile for the snakemake make system (which is included in the conda environment).
It will assemble the data and run all plotting scripts correctly. Simply run:
```bash
Snakemake
```
Editing the Snakefile is the simplest way of testing other drug combinations.
Simply add/remove output files from the __all__ rule at the top.
To get an idea of what the files contain, it is advisable to first run Snakemake once and examine the output.

__plots/diagnostics.py__  
Plotting functions that generate the Axitinib-supplementation plots.

__plots/reduction.py__  
Plotting functions that generate any of the dose-reduction or ic50 plots with asciminib.

__plots/triple.py__  
Hardcoded plotting function for the triple combination of Asciminib, Axitinib and Bosutinib.

__plots/add-compilation.py__  
Utility code that assembles drug info file (with kinetics information) and IC50's from a csv.

---

## Experiment analysis
Raw data and analysis in the form of a jupyter notebook is available in the expeirment folder. The notebook requires R and the rethinking package

https://github.com/rmcelreath/rethinking
