r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""



# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['h_dim'] = 512
    hypers['z_dim'] = 32
    hypers['x_sigma2'] = 0.001
    hypers['learn_rate'] = 1e-4
    hypers['betas'] = (0.5, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
The $\sigma^2$ hyperparameter measures the spread of the generated samples (data distribution) around the approximated mean value of the instance space. When the $\sigma^2$ is close to 0, it indicates that the data points tend to be close to the mean.

* A low value of $\sigma^2$ affects the results of the samples that are mostly similar to each other (closer to dataset images).
* A high value of $\sigma^2$ affects the results in a more extensive variety of the samples, less match to each other, and the model will be considered more visionary.
"""

part2_q2 = r"""
**Your answer:**
1. Reconstruction loss is the distance between the reconstructed image to the original. KL divergence loss measures the encoder probability function to the latent space prior.

2. In VAE, let X be the data we want to model, z be latent variable, P(X) be the probability distribution of data, P(z) be the probability distribution of the latent variable, and P(X|z) be the distribution of generating data given latent variable. While training our VAE, the encoder should try to learn the simpler distribution K(z|X) such that it is as close as possible to the actual distribution P(z|X). This is where we use KL divergence as a measure of a difference between two probability distributions. The VAE objective function thus includes this KL divergence term that needs to be minimized. Therefore, By minimizing the KL divergence loss term, we can result that the latent-space distribution and the instance-space distribution will be statistically independent.

3. The benefit is that we can adjust both distributions without affecting each other. The distribution parameters are tuned by different nets, which we train separately. Therefore, it makes our computations to become more stable and accurate.
"""

part2_q3 = r"""
**Your answer:**
By maximizing the evidence distribution, the generative model would produce images with better quality, since if we consider the latent space to be a normal distribution, the generative images that are closer to the original images are more likely to resemble the dataset.
"""

part2_q4 = r"""
**Your answer:**
The KL divergence has several mathematical meanings. Although it is used to compare distributions, it comes from the field of information theory, where it measures how much "information" is lost when coding a source using a different distribution other than the real one. 

Therefore, with regards to the KL divergence formula, the entropy is the measure of "information" in a source and generally describes how "surprised" you will be with the outcome of the random variable. 

One reason is the logarithm property of log⁡(xy)=log⁡(x)+log⁡(y), meaning the information of a source composed of independent sources (p(x)=p1(x)p2(x)) will have the sum of their information. This can only happen by using a logarithm.

Secondly, in most cases the standard deviation values are small. If we would try to use the log transform, we can map the smaller numbers onto a larger interval. This would grant greater stability to the Floating-Point arithmetic.
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['z_dim'] = 128
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.2
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['lr'] = 0.0002
    hypers['discriminator_optimizer']['betas'] = (0.5, 0.999)
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.0002
    hypers['generator_optimizer']['betas'] = (0.5, 0.999)
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
During the training phase, we're training both models, the discriminator model, and the generator model simultaneously, which lead us to perform sampling in two different cases:
* When we compute the generator's loss, we use the forward function on the sampled objects from the sampling function. Then, when we calculate the loss we will need those samples to be in the generator's computation graph. Hence, we maintain gradients when sampling from the GAN.
* When we compute the loss over the discriminator model, the gradients are irrelevant, since we use the sampled object for applying the discriminator path. Hence, we don't need to maintain gradients.
"""

part3_q2 = r"""
**Your answer:**
1. No, because if the discriminator isn't performing good results in the training phase, then the generator's loss can decrease below a given threshold, despite the fact that the generator model isn't performing well.
The reason is that the generator's loss might decrease when the generator is performing better results or when the discriminator is getting poor results.

2. It means that both the generator and the discriminator are performing good results. The generator produces more likely samples to the original images and the discriminator performs good classifications between the original and fake images.
"""

part3_q3 = r"""
**Your answer:**
GANs yield better results as compared to VAE. In VAE, we optimize the lower variational bound whereas in GAN, there is no such assumption. GANs don’t deal with any explicit probability density estimation, however, it performs a more complex loss function during the training phase. The failure of VAE in generating sharp images implies that the model is not able to learn the true posterior distribution, while GAN produces more creative images (despite the blurriness).
"""

# ==============
