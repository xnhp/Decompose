---
alias: zero-one max depth
---

![[ma-thesis/experiments/classification-zero_one-max_depth/line-plot.png]]

- RF classifiers (scikit-learn default)
- on MNISt - evaluated on 01-loss
- fixed ensemble size of $5$
# Notes
## 2023-07-07
- "sweet spot" at a certain depth, then risk increases again -- is this the notion of max depth as a regulariser? ^4defcb
- interesting to see how bias increases sharply again but seems to be somewhat mitigated by continuously increasing diversity
- this relationship would be nice to investigate
    - the individual models are worse, but because of "better" diversity (wasted votes etc?), we can still mitigate
        some individual performance?
## [[2023-07-10]] ^59843e
- note that this is the max depth hyperparam and constructs a new tree for each value of it, which is internally different from the one with the previous parameter vaue -> if we want to observe how output changes with additional levels (i.e. what does adding additional level really do), we should rather take trained trees and then cut off levels individually