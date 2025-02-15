# Penalizing Gradient Norm in Deep Learning

This is our work at ICML2022, which entitles "[Penalizing Gradient Norm for Efficiently Improving Generalization in Deep Learning](https://arxiv.org/abs/2202.03599)", by Yang Zhao, Hao Zhang and Xiuyuan Hu.

### 1. Short intro

##### 1.1 Overview

Basically, to penalize gradient norm in training, an additional term regarding the gradient norm $||\nabla_{\theta} L(\theta)||_2$ will be added on top of the empirical loss, which gives that,

$$\begin{aligned}
L(\theta) = L_{\mathcal{S}}(\theta) + \lambda ||\nabla_{\theta} L_{\mathcal{S}}(\theta)||_2
\end{aligned}$$

Gradient norm is considered a property that characterizes the flatness of the loss surface. Penalizing gradient norm motivates to bias the optimization to converge to flatter minima, leading to better model generalization. 

##### 1.2 Practical gradient computation of gradient norm

Based on the chain rule, the gradient of gradient norm is,

$$\begin{aligned}
\nabla_{\theta} L(\theta) = \nabla_{\theta} L_{\mathcal{S}}(\theta) + \lambda \cdot \nabla_{\theta}^2 L_{\mathcal{S}}(\theta) \frac{\nabla_{\theta} L_{\mathcal{S}}(\theta)}{||\nabla_{\theta} L_{\mathcal{S}}(\theta)||}
\end{aligned}$$

Computing the gradient of this gradient norm term in a straightforward way will involve the full computation of Hessian matrix. We introduce Taylor expansion to approximate the multiplication between the Hessian matrix and vectors, giving that

$$\begin{split}
    \nabla_{\theta} L(\theta) & = \nabla_{\theta} L_{\mathcal{S}}(\theta) + \lambda \cdot (\frac{\nabla_{\theta}L_\mathcal{S}(\theta +r\frac{\nabla_{\theta}L_{\mathcal{S}}(\theta)}{||\nabla_{\theta}L_{\mathcal{S}}(\theta)||}) - \nabla_{\theta}L_\mathcal{S}(\theta)}{r}) \\
    & = (1 - \frac{\lambda}{r}) \nabla_{\theta} L_{\mathcal{S}}(\theta) + \frac{\lambda}{r} \cdot \nabla_{\theta}L_\mathcal{S}(\theta +r\frac{\nabla_{\theta}L_{\mathcal{S}}(\theta)}{||\nabla_{\theta}L_{\mathcal{S}}(\theta)||})
\end{split}$$
 
where is a small scalar value. So, we need to set two parameters for gradient norm penalty, one for the penalty coefficient &\lambda& and the other one for $r$. And we will further use &\alpha = \lambda / r&. In particular, [SAM](https://github.com/google-research/sam) is a special implementation of this scheme, where $\alpha$ will always set equal to 1.

### 2. Training using this repo

- This repo is realized via the JAX framework. One should start by building the python environment listed in the requirements.txt file. 
- Training recipes are the .sh files in the Recipe folder and training configs are in the config folder. Training recipes will overwrite the involved configs. 
- 




### 3. Some results



If you find this helpful, you could cite the paper as

