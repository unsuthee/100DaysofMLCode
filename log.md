
# Day 1
I implemented a vanilla GANs and trained the model on MNIST dataset. I plotted the loss from discriminator and generator as well as displayed the images sampled from the generator at 10, 50, 100, 150, and 200 epochs.

- [blog](https://sutheeblog.wordpress.com/2018/08/14/day-1-vanilla-gans/)
- [code](https://github.com/unsuthee/100DaysofMLCode/tree/master/day1)

# Day 2
I studied Indepedent Component Analysis (ICA) because I like the idea of changing variable technique used for solving ICA. The idea of changing variable has also recently been used again in RealNVP and Normalizing Flows. It is a good idea to pick up the basic from ICA first before exploring these advance techniques.

- [blog](https://sutheeblog.wordpress.com/2018/08/17/day-2-independent-component-analysis-ica/)

# Day 3
I plan to implement some algorithms to solve ICA and compare my algorithms with the skLearn's fastICA. It turns out to be very tricky to implement ICA based on the maximum likelihood approach. I encountered many numerical instability, most comes from computing the log of determinant. 

- [blog](https://sutheeblog.wordpress.com/2018/08/19/day-3-ica-with-gradient-ascent/)
- [code](https://github.com/unsuthee/100DaysofMLCode/blob/master/day2/PlayWithICA.ipynb)

# Day 4
I worked on Neural Autoregressive model and trained the model on MNIST dataset. I sampled a few images from the learned distribution but the results are not good at all.

- [blog](https://sutheeblog.wordpress.com/2018/08/20/day-4-nade-revisit/)
- [code](https://github.com/unsuthee/100DaysofMLCode/tree/master/day4)

# Day 5
I worked on Mask Autoencoder which is an improvement of NADE model. I used the pyTorch implementation by Karpathy and be able to generate some MNIST digits. The results are not good but slightly better than NADE.

- [blog](https://sutheeblog.wordpress.com/2018/08/22/day-5-made-mask-autoencoder/)

# Day 6
I worked on DCGAN. This model uses deconvolutional layers as a generator and convolutional layers as a discriminator. I implemented DCGAN and be able to generate some MNIST digits. The results look good and are much better than vanilla GAN.

- [blog](https://sutheeblog.wordpress.com/2018/08/22/day-6-dcgan/)
- [code](https://github.com/unsuthee/100DaysofMLCode/blob/master/day6/run_DCGAN.py)

# Day 7
I worked on Conditional VAE. I use convolution and deconvolution as part of the encoder and decoder. After applying the KL annealing, the generated images are reasonably good. 

- [blog](https://sutheeblog.wordpress.com/2018/08/23/day-7-conditional-vae/)
- [code](https://github.com/unsuthee/100DaysofMLCode/tree/master/day7)

# Day 8
I trained CVAE and DCGAN on FashionMNIST and EMNIST. The results are not good. 
- [blog](https://sutheeblog.wordpress.com/2018/08/24/day-8-move-away-from-mnist-datasets/)
