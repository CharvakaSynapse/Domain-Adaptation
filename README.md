# Office-31 Domain Adaptation with CDAN and Semi-Supervised Enhancements

This notebook demonstrates progressive improvements in **unsupervised domain adaptation** on the Office-31 dataset using PyTorch, from the CDAN baseline to advanced semi-supervised techniques.

---

## 📚 What’s Inside

- **CDAN (Conditional Domain Adversarial Network)** baseline
- **Class-Balanced Curriculum Pseudo-Labeling** for target domain
- **Consistency Regularization** with strong/weak augmentation and KL loss

All code, explanations, and results are contained in the notebook:  
> **`DANN_DA_FINAL.ipynb`**

---

## 📈 Results: Amazon → DSLR

| Method                                                            | Test Accuracy (%) |
|-------------------------------------------------------------------|-------------------|
| CDAN Baseline                                                     | 86.32             |
| CDAN + Class-Balanced Curriculum Pseudo-Labeling                  | 87.39             |
| CDAN + Pseudo-Labeling + Consistency Regularization (Strong/Weak) | **89.10**         |

---

## 📝 Method Highlights

- **CDAN**: Aligns source and target domains with a conditional domain discriminator.
- **Pseudo-Labeling**: Uses curriculum learning and class balancing to safely leverage unlabeled target data.
- **Consistency Regularization**: Enforces prediction agreement between strong and weak augmentations of target images (inspired by FixMatch).

**Total loss:**
\[
L_{total} = L_{cls} + \lambda L_{domain} + \eta L_{entropy} + \beta L_{pseudo} + \gamma L_{cons}
\]
where each term encourages robust, transferable representations.

---

## ⚡ How to Run

1. Download the [Office-31 dataset](https://faculty.cc.gatech.edu/~judy/domainadapt/).
2. Organize it as described in the notebook.
3. Launch `DANN_DA_FINAL.ipynb` and follow the step-by-step instructions.

---

## 🔗 References

- [Conditional Adversarial Domain Adaptation (CDAN), NeurIPS 2018](https://arxiv.org/abs/1705.10667)
- [FixMatch: Semi-Supervised Learning, NeurIPS 2020](https://arxiv.org/abs/2001.07685)

---

*For questions or suggestions, feel free to open an issue or reach out!*
