# wls-spe-gurobi

## Weighted Least Squares State Estimation for Power Grids (pandapower-based)

This repository provides an implementation of **Weighted Least Squares (WLS) State Estimation** for electrical power grids modeled using [**pandapower**](https://www.pandapower.org/). The primary contribution of this work is a custom integration of WLS State Estimation into the pandapower framework, allowing users to perform realistic and scalable state estimation directly on pandapower networks. The implementation focuses on clarity, modularity, and extendability—suitable for both academic use and practical experimentation.

### Future Work

This project sets the foundation for for parameter estimation methods:

- **Parameter Estimation** based on:
  - **State Vector Augmentation** (Joint): Integrating unknown parameters into the state vector for joint estimation.
  - **Method of Residual Analysis** (Recursive): Using residuals from state estimation to identify model inaccuracies or bad data.

Install dependencies using:

```bash
pip install -r requirements.txt 
```
