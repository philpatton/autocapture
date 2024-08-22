# Simulating capture histories, and estimating paramters, under misidentification

Code to recreate analysis *Evaluating tradeoffs between automation and bias in population assessments relying on photo-identification.* 

## Simulation

The simulation proceeds in three general steps.

1. Simulate capture histories using a POPAN model. 
2. Corrupt capture histories with misidentification error, using predetermined rates.
3. Estimate demographic parameters by fitting capture-recapture models to the corrupted histories.

### Configuration

Users can provide their own configs (TODO: Improve documentation for this process). To recreate the results from the paper, run the following commands. This will create `.yaml` files in the the `config` directory. 

```
python -m src.config 
```

### Simulating capture histories

The `src.simulate` script accomplishes tasks #1 and #2 above for a given strategy. The following command runs the simulate script.

```
python -m src.simulate --strategy check_0
python -m src.simulate --strategy check_5
...
python -m src.simulate --strategy check_25

```

The script uses the `POPAN` class, from the `model` module, and its `simulate()` function to simulate a capture history. Then it corrupts the history with misidentifications, using the `MissID` class from the `miss_id` module. By default, `simulate` simulates 100 replicates for dataset in a given strategy. 

### Estimating demographic parameters

The `src.estimate` script accomplishes task #3 above for a given scenario and model. It relies on the `POPAN` class, from the `model` module, and its `estimate()` function. 

```
python -m src.estimate --strategy check_0
python -m src.estimate --strategy check_5
...
python -m src.estimate --strategy check_25

```

The parameters are estimated using PyMC. 