# Simulating capture histories, and estimating paramters, under misidentification

Code to recreate analysis *Evaluating tradeoffs between automation and bias in population assessments relying on photo-identification.* 

## Simulation

The simulation proceeds in three general steps.

1. Simulate capture histories using a POPAN model. 
2. Corrupt capture histories with misidentification error, using predetermined rates.
3. Estimate demographic parameters by fitting capture-recapture models to the corrupted histories.

### Configuration

Users should have some form of conda installed (I prefer [miniforge](https://github.com/conda-forge/miniforge)). Then, they need to set up a conda environment using the `requirements.txt` file. For example,

```
conda create --name <env> --file <this file>
```

The modules rely on config files, in the form of `.yaml`. Users can provide their own configs (TODO: Improve documentation for this process). Otherwise, they can recreate the results from the paper using the `scr.config` script, which creates `.yaml` files in the the `config` directory. The `scr.config` script is called from the command line, for example,

```
python -m src.config 
```

### Simulating capture histories

The `src.simulate` script simulates data under an open population capture recapture model (by default, a POPAN model, although the CJS model is available too). Then, it corrupts the capture histories using the misidentification process. Users can run the script from the command line, specifying the strategy with the `--strategy` argument,

```
python -m src.simulate --strategy check_0
python -m src.simulate --strategy check_5
...
python -m src.simulate --strategy check_25

```

The script uses the `POPAN` class, from the `model` module, and its `simulate()` function to simulate a capture history. Then it corrupts the history with misidentifications, using the `MissID` class from the `miss_id` module. By default, `simulate` simulates 100 replicates for dataset in a given strategy. 

### Estimating demographic parameters

The `src.estimate` script accomplishes task #3 above for a given scenario and model. It relies on the `POPAN` class, from the `model` module, and its `estimate()` function. Users can run the script from the command line, specifying the strategy with the `--strategy` argument,

```
python -m src.estimate --strategy check_0
python -m src.estimate --strategy check_5
...
python -m src.estimate --strategy check_25

```

The parameters are estimated using PyMC. 

### Collating the results

Finally, users can collate the results into a .csv file, which contains relevant statistics from the posterior distribution of each parameter. To do so, Users can run the script from the command line, specifying the strategy with the `--strategy` argument,

```
python -m src.analyze --strategy check_0 
```