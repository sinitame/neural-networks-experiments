# Batch normalisation

One of the main assumption made when training learning systems is to supposed that the distribution of the inputs stays the same throughout training. For linear models, which simply maps input data to some appropriate outputs, this condition is always satisfied but it is not the case when dealing with Neural Networks which are composed of several layers which are stack on top of each other. 

In such architecture, each layers inputs are affected by the parameters of all preceding layers (small changes to the network parameters amplify as the network becomes deeper). As a consequence, a small change made during the backpropagation step within a layer can produce a huge variation of the inputs of another layer and at the end change their distribution. During the training, each layers need to continuously adapt to the new distribution obtained from the previous one and this slows down the convergence.

Batch normalization [1] overcome this issue and make the training more efficient at the same time by reducing covarience shift within internal nodes during the course of training and with the advantages of working with batches.

## 1. Reduce internal covariance shift via mini-batch statistics

One way reduce remove the ill effects of the internal covariate shift within a Neural Network is to normalize layers inputs. This operation not only enforce inputs to have the same distribution but also whiten each of them. This method is motivated by some studies [3,4] showing that the network training converges faster if its inputs are whitened and as a consequence, enforcing the whithening of the inputs of each layers is a desirable property for the network.

The full whitening of each layer’s inputs is costly and not everywhere differentiable. Batch normalization overcome these issue by considering two assumptions:

- instead of whitening the features in layer inputs and outputs jointly, we will normalize each scalar feature independently, by making it have the mean of zero and the variance of 1.
- Instead of using the entire dataset to normalize activations, we use mini-batches as *each mini-batch produces estimates of the mean and variance* of each activation.

For a layer with d-dimentional inputs x = $(x^{(1)} ... x^{(d)})$ we obtain the normalization with the following formulae (with expectation and variance computed over a batch B): 
$$
\hat{x}^{(k)} = \frac{x^{(k)} - \text{E}[x^{(k)}]_{\mathcal{B}}}{\sqrt{\text{Var}[x^{(k)}]_{\mathcal{B}}}}
$$


However, simply normalizing each input of a layer may change what the layer can represent. For instance, normalizing the inputs of a sigmoid would constrain them to the linear regime of the nonlinearity. Such a behavior is not desirable for the network as it will reduce his representative power (it would become equivalent of a single single layer network).

(simoid image avec highlight sur la partie non linéaire )

To address this, batch normalization also ensure that the transformation inserted in the network can represent the identity transform (the model still learn some parameters at each layers that adjust the activations recieved from the previous layer without linear mapping) . This is accomplished by introducing a pair of learnable parameters gamma_k and beta_k  which scale and shift the nomalized value according to what the model learns.

At the end, the resulting layers inputs (based on previous layer outputs x) are given by:
$$
y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}
$$


## 2. Batch normalization algorithm



### During training

```python
def training_batch_norm(X, gamma, beta, eps = 1e-5):
    if len(X.shape) not in (2, 4):
        raise ValueError("only support dense or 2dconv")
    
    # dense layer
    elif len(X.shape) == 2:
        mean = torch.mean(X, axis=0)
        variance = torch.mean((X-mean)**2, axis=0)
        X_hat = (X-mean) * 1.0 /torch.sqrt(variance + eps)
        out = gamma * X_hat + beta
        
    elif len(X.shape) == 4:
        N, C, H, W = X.shape
        mean = torch.mean(X, axis = (0, 2, 3))
        variance = torch.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
        X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / torch.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))
        
    return out
```



### During inference

## 3. Experiments on MNIST

![](imgs/training-loss.png)

![](imgs/activation-distribution.png)

## Conclusion

### Advantages of using batch normalisation for training

- The gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases.
- The computation over a batch size can be much more efficient than $m$ computations for individual examples due to the parallelism afforded by GPUs.
- ...

## References

[1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *arXiv preprint arXiv:1502.03167* (2015).

[2] Shimodaira, Hidetoshi. "Improving predictive inference under covariate shift by weighting the log-likelihood function." *Journal of statistical planning and inference* 90.2 (2000): 227-244.