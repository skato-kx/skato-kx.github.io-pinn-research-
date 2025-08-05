# Physics-Informed Neural Networks for Steady-State Flow

Implementation and summary of the PINN method for solving PDEs using deep learning.

## 🔗 Paper Info

- **Title**: [Physics-Informed Neural Networks for Steady-State Flow](https://arxiv.org/abs/xxxx.xxxxx)
- **Authors**: X et al.
- **Keywords**: PINN, PDE, Deep Learning, Physics-based ML

## 🧠 What is PINN?

A physics-informed neural network (PINN) is a neural network trained not only on data but also on physics equations (PDEs) by including their residuals in the loss function.

## 📝 Summary of the Paper

- The authors propose ...
- The model architecture includes ...
- The loss function consists of ...

## 🛠️ Implementation

This repo contains:
- `train.py`: Training loop using autograd
- `model.py`: PINN architecture
- `pde.py`: Definition of the PDE residuals
- `utils/`: Helper functions

## 📊 Results

- Boundary conditions satisfied with MAE < 0.01
- Visualization of prediction vs ground truth:

![sample](./assets/pinn_result.png)

## 🗒️ To Do

- [ ] Try different activation functions
- [ ] Test on different PDEs (e.g. Burgers' equation)
- [ ] Hyperparameter tuning

## 💬 Thoughts

> "I found the paper surprisingly easy to implement but tricky to stabilize. The way they handle boundary conditions with soft constraints is really elegant."

## 📎 Related Links

- [Original Paper](https://arxiv.org/abs/xxxx.xxxxx)
- [My other implementations](https://github.com/skato)
