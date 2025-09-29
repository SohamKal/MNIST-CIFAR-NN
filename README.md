# MNIST-CIFAR-NN

Small, self-contained deep learning project showing three core skills:

1) building and training a **Logistic Regression** classifier on **MNIST**,  
2) implementing a lightweight **Fully-Connected Neural Network (FNN)** for **CIFAR-10**, and  
3) running a simple **hyper-parameter search** loop (learning rate, weight decay, batch size) with early stopping.


---

## Highlights

- **MNIST (LogReg)**: **92.0%** test accuracy (0.9201) on a T4 GPU in ~3–4 minutes.  
- **CIFAR-10 (FNN)**: compact 2-hidden-layer baseline implemented from scratch (no autotrainers).  
- **Tuning**: grid search over LR / weight decay/batch size with early stopping on validation accuracy.  
- **Clean structure**: training, validation, and testing are separate, with clear entry points.

---

## What I built

### 1) Logistic Regression (MNIST)
- Model: single `nn.Linear(28*28 -> 10)` with `CrossEntropyLoss`
- Data: standard torchvision MNIST with normalization
- Training: SGD (momentum-free), weight decay, ~15 epochs (tuning uses early stop)
- Result snapshot:
  - **Accuracy**: **92.01%**  
  - **Run time**: ~206s on a T4 GPU (varies by hardware)

### 2) FNN Baseline (CIFAR-10)
- Architecture: `Flatten → Linear(3072→64) → Tanh → Linear(64→32) → ReLU → Linear(32→10)`  
- Loss: Cross-Entropy; Optimizers: SGD/Adam during sweeps  
- Purpose: Provide a clear, minimal non-convolutional baseline before moving to CNNs

### 3) Hyper-parameter Search
- Grid over: **learning rate**, **weight decay**, **batch size** (for both MNIST LogReg and CIFAR-10 FNN)
- Shared training/validation routine, **early stopping**, and metric tracking
- Prints best params + validation accuracy

---

