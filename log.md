
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
