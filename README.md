# oxynet

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_oxynet&metric=alert_status)](https://sonarcloud.io/dashboard?id=durbin-164_oxynet)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_oxynet&metric=coverage)](https://sonarcloud.io/dashboard?id=durbin-164_oxynet)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_oxynet&metric=bugs)](https://sonarcloud.io/dashboard?id=durbin-164_oxynet)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_oxynet&metric=code_smells)](https://sonarcloud.io/dashboard?id=durbin-164_oxynet)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_oxynet&metric=ncloc)](https://sonarcloud.io/dashboard?id=durbin-164_oxynet)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_oxynet&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=durbin-164_oxynet)

This is a toy Deep Learning Framework from scratch. If you want to learn how to automatic differentiation work, see this code. This is under developing.
For start see example folder and tests folder to understand functionality.

## Variable
### Tensor
Tensor is the main auto differentiable multidimensional variable. It supports almost all frequently used function such That
#### Creation

```python
t1 = Tensor(10, requires_grad=True)
t2 = Tensor([1, 2, 3], requires_grad=True)
t3 = Tensor([[1, 2, 3],[4, 5, 6]], requires_grad=True)
```

