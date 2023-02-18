# Code for EURING (2023) presentation

Simulate data for a scenario in the `config_path`:
```python -m simulate --config_path config/dry-run.yaml --experiment_name dry-run```

Estimate the model parameters for those data
```python -m estimate --config_path config/dry-run.yaml --experiment_name dry-run```


## TODO items 

1. Write tests for estimate
2. Write configs for all scenarios (40 catalogs by 3 levels of oversight). 
