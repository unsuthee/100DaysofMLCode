# 100DaysofMLCode
This is my commitment to the 100 days ML coding chellenge proposed by Siraj Raval. 

# Topic of studies
My focus is Deep Genenerative Model including:
- Adversarial Training
- Variational Autoencoder, ELBO methods
- Autoregressive Model and architectures
- Invertible architecture, exact inference methods

**Day1**: I implemented a vanilla GANs and trained the model on MNIST dataset. I plotted the loss from discriminator and generator as well as displayed the images sampled from the generator at 10, 50, 100, 150, and 200 epochs.

- [blog](https://sutheeblog.wordpress.com/2018/08/14/day-1-vanilla-gans/)
- [code](https://github.com/unsuthee/100DaysofMLCode/tree/master/day1)

**Day2** I studied Indepedent Component Analysis (ICA) because I like the idea of changing variable technique used for solving ICA. The idea of changing variable has also recently been used again in RealNVP and Normalizing Flows. It is a good idea to pick up the basic from ICA first before exploring these advance techniques.

- [blog](https://sutheeblog.wordpress.com/2018/08/17/day-2-independent-component-analysis-ica/)

**Day3** I plan to implement some algorithms to solve ICA and compare my algorithms with the skLearn's fastICA. It turns out to be very tricky to implement ICA based on the maximum likelihood approach. I encountered many numerical instability, most comes from computing the log of determinant. 

- [blog](https://sutheeblog.wordpress.com/2018/08/19/day-3-ica-with-gradient-ascent/)
- [code](https://github.com/unsuthee/100DaysofMLCode/blob/master/day2/PlayWithICA.ipynb)

**Day4** I worked on Neural Autoregressive model and trained the model on MNIST dataset. I sampled a few images from the learned distribution but the results are not good at all.

- [blog](https://sutheeblog.wordpress.com/2018/08/20/day-4-nade-revisit/)
- [code](https://github.com/unsuthee/100DaysofMLCode/tree/master/day4)

**Day5** I worked on Mask Autoencoder which is an improvement of NADE model. I used the pyTorch implementation by Karpathy and be able to generate some MNIST digits. The results are not good but slightly better than NADE.

- [blog](https://sutheeblog.wordpress.com/2018/08/22/day-5-made-mask-autoencoder/)

**Day6** I worked on DCGAN. This model uses deconvolutional layers as a generator and convolutional layers as a discriminator. I implemented DCGAN and be able to generate some MNIST digits. The results look good and are much better than vanilla GAN.

- [blog](https://sutheeblog.wordpress.com/2018/08/22/day-6-dcgan/)
- [code](https://github.com/unsuthee/100DaysofMLCode/blob/master/day6/run_DCGAN.py)

**Day7** I worked on Conditional VAE. I use convolution and deconvolution as part of the encoder and decoder. After applying the KL annealing, the generated images are reasonably good. 

- [blog](https://sutheeblog.wordpress.com/2018/08/23/day-7-conditional-vae/)
- [code](https://github.com/unsuthee/100DaysofMLCode/tree/master/day7)

# Day 8
I trained CVAE and DCGAN on FashionMNIST and EMNIST. The results are not good. 
- [blog](https://sutheeblog.wordpress.com/2018/08/24/day-8-move-away-from-mnist-datasets/)

# Day 9
I trained CVAE and DCGAN on CIFAR10. The results are okay after a few trial-and-errors on the model architecture.
- [blog](https://sutheeblog.wordpress.com/2018/08/29/day-9-dcgan-and-cvae-on-cifar10/)

# Day 10
I worked on conditional GAN. My implemenation of CGAN did not work as I encountered the mode collapsing problem.
- [blog](https://sutheeblog.wordpress.com/2018/09/05/day-10-mode-collapsing-on-my-cgan/)

# Day 11
My CGAN works now. I use a 2D one-hot representation to represent a class label. This idea works well.
- [blog](https://sutheeblog.wordpress.com/2018/09/05/day-11-2d-one-hot-representation/)

# Day 12
I worked on a few techniques for handling a discrete output from the encoder of VAE or the generator of GANs. The first approach is to expand the expectation term in VAE. I rewrote semi-supervised VAE.
- [blog](https://sutheeblog.wordpress.com/2018/09/12/day-12-handling-discrete-output-in-vae/)

# Day 13
I completed my implemention of semi-supervised VAE. 
- [blog](https://sutheeblog.wordpress.com/2018/09/13/day-13-implementation-of-semi-supervised-vae/)

# Day 14
Semi-supervised VAE with Gumbel-Softmax

# Day 15
Semi-supervised VAE with policy gradient method
