# Code for Patton et al. (TBD) 

Code to recreate analysis in Patton et al. (TBD) *Mitigating downstream effects of automated photo--identification on capture--recapture estimators.* 

## Simulation

The simulation proceeds in three general steps.

1. Simulate true histories using a POPAN model of population dynamics (`src.popan.POPAN().simulate()`).
2. Simulate capture histories by corrupting true histories with misidentification errors (`src.miss_id.MissID`).
3. Estimate demographic parameters by training capture-recapture models (`src.popan.POPAN().estimate()`).

### Configuration

Before tackling any of the above tasks, you may need to write the configs. To do so, use `src.config` script, specifying a scenario. The scenario dictates the error rate for a given catalog. 

```
python -m src.config --scenario fully
python -m src.config --scenario semi5
```

### Simulating capture histories

The `src.simulate` script accomplishes tasks #1 and #2 above for a given scenario. In the example below, we simulate data for two scenarios: "fully" and "semi5"

```
python -m src.simulate --scenario fully
python -m src.simulate --scenario semi5
```

The script uses the `src.popan.POPAN` class and its `simulate()` function for task #1. For task #2, it uses the `src.miss_id.MissID` class. By default, simulates 100 replicates for each of the 39 catalogs in a given scenario. 

### Estimating demographic parameters

The `src.estimate` script accomplishes task #3 above for a given scenario and model. It relies on the `src.popan.POPAN()` class and its `estimate()` function. 

```
python -m src.estimate --scenario fully --model cjs
python -m src.estimate --scenario semi5 --model cjs
```

Models are trained using PyMC code adapted from [Austin Rochford](https://austinrochford.com/posts/2018-01-31-capture-recapture.html) 
