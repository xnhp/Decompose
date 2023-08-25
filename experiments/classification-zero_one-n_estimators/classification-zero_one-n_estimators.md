---
alias: classification-zero_one n_estimators
---
![Plot](plot.png)

- RF (scikit-learn default) classifier on MNIST
- Evaluated with 01-Loss
- number of trees from 1 to 20
Plotted:
- [[Bias-Variance-Diversity-Effect Decomposition]]
- [[Member Deviation]]
- [[Expected Member Loss]]
# Notes
## 2023-06-10
- entire consideration should apply to any bagging ensemble, not limited to trees/forests
- recall that `member-deviation` is an upper-bound for diversity-effect (triangle inequality)
- interesting that `member-deviation` grows faster than overall diversity-effect
## 2023-06-22
- expected average member loss stays constant with ensemble size, as expected. nothing surprising happening here.
