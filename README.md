# Code for Patton et al. (TBD) 

Code to recreate analysis in Patton et al. (TBD) *Mitigating downstream effects of automated photo--identification on capture--recapture estimators.* 

## Simulation

The simulation proceeds in three general steps.

1. Simulate true histories using a POPAN model of population dynamics (`src.popan.POPANSimulator`).
2. Simulate capture histories by corrupting true histories with misidentification errors (`src.miss_id.MissID`).
3. Estimate demographic parameters by training capture-recapture models (`src.popan.POPANEstimator`; `src.cjs.CJSEstimator`).

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

The script calls `src.popan.POPAN()` (task #1) and `src.miss_id.MissID()` (task #2) for a given trial, simulating 100 trials for each of the 40 catalogs in a given scenario. 

### Estimating demographic parameters

The `src.estimate` script accomplishes task #3 above for a given scenario and model. There are two possible models: POPAN (`src.popan.POPANEstimator`) and CJS (`src.cjs.CJSEstimator`).

```
python -m src.estimate --scenario fully --model cjs
python -m src.estimate --scenario semi5 --model cjs
```

Models are trained using PyMC code adapted from [Austin Rochford](https://austinrochford.com/posts/2018-01-31-capture-recapture.html) 
