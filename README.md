# Code for EURING (2023) presentation

There are two primary scripts for recreating the analysis, `src.simulate` and `src.estimate`. For each, the `scenario` must be specified. For example, simulate data for both scenarios.

```
python -m src.simulate --scenario fully
python -m src.simulate --scenario semi5
```

This script relies on two classes: `src.popan.POPANSimulator` and `src.miss_id.MissID.` `POPANSimulator` simulates data for a Jolly-Seber model. `MissID` adds misidentification errors to a capture history. 

```
python -m src.estimate --scenario fully
python -m src.estimate --scenario semi5
```

`src.estimate` uses `src.popan.POPANEstimator` for model fitting, based on a PyMC model adapted from [Austin Rochford](https://austinrochford.com/posts/2018-01-31-capture-recapture.html) 

You may need to write the configs before anything else. To do so, use `src.config` script.

```
python -m src.config --scenario fully
python -m src.config --scenario semi5
```
