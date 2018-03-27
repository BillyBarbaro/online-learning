# online-learning

I've been doing machine learning research in the area of
online classification.  As great as the Python tools are
for batch learning, there seems to be a real lack of
online learning libraries, so I decided I would
contribute my implementations of several online learning
algorithms I've been using in my research as a start
at building a Python library for this type of machine learning.

I hope to build this out into into a full library for online learning,
so if you're interested in helping out or have any ideas for
the project, don't hesitate to get in touch!  The starting work
here is based off of the Matlab and C++ implementations found in
[LIBOL](https://github.com/LIBOL).  The following alogithms
are currently included:

1. **Passive Aggressive Algorithms**
    * [Online Passive-Aggressive Algorithms](http://www.jmlr.org/papers/v7/crammer06a.html)
2. **Random Budgeted Perceptron**
    * [Tracking the best hyperplane with a simple budget Perceptron](https://link.springer.com/article/10.1007/s10994-007-5003-0)
3. **Fourier Online Gradient Descent**
    * [Large Scale Online Kernel Learning](http://jmlr.org/papers/v17/14-148.html)
4. **Dual Space Gradient Descent**
    * [Dual Space Gradient Descent for Online Learning](http://papers.nips.cc/paper/6560-dual-space-gradient-descent-for-online-learning)