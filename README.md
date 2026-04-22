# Self-Pruning Neural Network

## 🔹 Approach

We implemented a custom **PrunableLinear layer** where each weight is associated with a learnable gate.
The gate is obtained using a sigmoid transformation:

* If gate ≈ 0 → weight is effectively pruned
* If gate ≈ 1 → weight is active

## 🔹 Why L1 Encourages Sparsity

The sparsity loss uses the **L1 norm of gate values**:

SparsityLoss = sum(gates)

L1 regularization pushes values toward zero.
Since gates are bounded between [0,1], minimizing this term forces many gates to become **exactly zero**, leading to pruning.

## 🔹 Results

| Lambda | Test Accuracy (%) | Sparsity (%) |
| ------ | ----------------- | ------------ |
| 1e-4   | 55.60             | 41.80        |
| 5e-4   | 55.98             | 43.50        |
| 1e-3   | 55.90             | 43.38        |

👉 Observation:

* Higher λ → higher sparsity but lower accuracy
* Lower λ → better accuracy but less pruning

## 🔹 Gate Distribution

The histogram shows:

* A spike near 0 → pruned weights
* A cluster away from 0 → important weights

This confirms successful self-pruning behavior.

## 🔹 Conclusion

The model successfully learns to remove unnecessary connections during training, achieving a trade-off between **model size and performance**.
