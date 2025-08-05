---
layout: default
title: PINN Summary
---

# Physics-Informed Neural Networks

As a first step in learning machine learning techniques related to physics, I decided to read this foundational and original paper on Physics-Informed Neural Networks (PINNs). Its implementation is relatively simple, making it a good starting point for beginners. In this post, I’ll walk through how PINNs integrate physical laws into neural networks, from the perspective of someone new to the field.

## 🔗 Paper Info

- **Title**: [Physics-Informed Neural Networks for Steady-State Flow](https://arxiv.org/abs/xxxx.xxxxx)
- **Authors**: X et al.
- **Keywords**: PINN, PDE, Deep Learning, Physics-based ML

## 🧠 What is NN?
A neural network is generally a system that learns patterns from past data in order to make predictions or classifications. More specifically, it learns the relationship between an input 𝑥 and an output 𝑢 by observing many examples. During training, the network predicts a value of 𝑢 for a given 𝑥, compares it to the true (ground truth) value, and then adjusts its internal parameters—across many interconnected neurons—to minimize the error between the prediction and the truth. This iterative process is what constitutes the typical learning mechanism of a neural network.

## 🧠 Why PINN?
However, when the target of prediction is a complex physical phenomenon, it's often difficult—or even impossible—to obtain the true values of 𝑢. In such cases, there may not be enough data available, making it hard for a standard neural network to learn effectively. This is where Physics-Informed Neural Networks (PINNs) come in. Instead of relying solely on data, PINNs incorporate physical laws—specifically, partial differential equations (PDEs)—into the training process, allowing the network to learn in a more principled and constrained way. 

## 🔍 What is a PDE?

In the context of PINNs, a **partial differential equation (PDE)** represents a fundamental physical law—one that governs the behavior of systems in our universe. These equations are considered to hold true at all times, regardless of the specific situation.

That said, the **form of the PDE stays the same**, but the **functions contained within it—especially the solution function \( u \)**—**depend on the specific physical scenario**. The role of the PINN is to approximate this function \( u \), which varies from case to case.

To illustrate with a simple analogy: the equation \( F = ma \) is always valid as a law of motion. But the specific values of \( F \), \( m \), and \( a \) depend on the object or system being studied. Similarly, in a PDE, the general law is fixed, but the solution function \( u \) changes depending on the particular problem.

From a beginner's perspective, this idea can be a little confusing—I also found myself stumbling over it at first.  
But once you realize that the PDE itself provides the **rules**, and \( u \) is the **unknown function you're trying to find**, the overall framework of PINNs becomes much clearer.

## Two types of approaches
PINNs can be broadly categorized into two types of approaches depending on how they treat time: continuous-time models and discrete-time models.
Below, I’ll briefly explain how each of them works.

## Continuous-Time Models
In this approach, the governing PDE can be rewritten into the general form:
ut + N [u; λ] = 0
where 𝑢 is the solution, 𝑥 and 𝑡 are the space and time variables, and 𝜆 represents known parameters. All components of this equation—time derivatives, spatial derivatives, and other terms—can be expressed in terms of the predicted function u(x,t) and its derivatives.

While this might sound abstract at first, the key idea is simple: since every PDE is, by definition, an equation, we can always rearrange it to isolate everything on one side and set the equation equal to zero—i.e., something = 0. This "something" can be calculated as long as we know the input variables 𝑥 and 𝑡, and the predicted output 𝑢 from the neural network.

So, by feeding arbitrary values of 𝑥 and 𝑡 into the network and computing this residual term, we can use it directly as part of the loss function. In other words, the network is trained not by minimizing the difference between predicted and ground-truth data, but by minimizing the violation of the physical law itself. This allows us to train a neural network without requiring labeled data, as long as we know the governing PDE.

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
