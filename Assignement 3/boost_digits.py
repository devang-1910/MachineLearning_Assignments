import numpy as np
from scipy.io import *
import matplotlib.pyplot as plt
from utils import *

# boosting for recognizing MNIST digits

# Load the data X and labels t
data=loadmat('digits.mat')
X,t=data['X'],data['t']
t=t.astype(int)

# X is 28x28x1000, t is 1000x1
# Each X[:,:,i] os a 28x28 image

# Subsample images to be 14x14 for speed
X=X[::2,::2,:]

# Set up target values
# 4s are class +1, all others are class -1
f4=(t==4)
n4=(t!=4)
t[f4]=1
t[n4]=-1

# 14,14,1000
nx,ny,ndata = X.shape

# Number to use as training images
ntrain = 500

# Training and test images
X_train = X[:,:,:ntrain]
t_train = t[:ntrain]
X_test = X[:,:,ntrain:]
t_test = t[ntrain:]

# Boosting code goes here
niter = 100

# Initialize the weights
weights = np.ones(ntrain)/ntrain
classifier = {'alpha':np.zeros(niter), 'd':np.zeros((niter,2)).astype(int), 'p':np.zeros(niter), 'theta':np.zeros(niter)}

for iter in range(niter):

    alpha = 0
    # Find the best weak learner
    # Find the best weak learner
    d, p, theta, correct = findWeakLearner(X_train, t_train, weights)

    # Compute weighted error
    epsilon = np.sum(weights[~correct])
    epsilon = epsilon / (np.sum(weights) + np.finfo(float).eps)

    # Compute alpha
    alpha = 0.5 * np.log((1 - epsilon) / (epsilon + np.finfo(float).eps))

    # Update weights
    hypothesis = np.where(correct, 1, -1)
    weights = weights * np.exp(-alpha * t_train.squeeze() * hypothesis)
    weights = weights / (np.sum(weights) + np.finfo(float).eps)

    ###########################################################
    ##################### End fill in #########################
    ###########################################################
    
    classifier['alpha'][iter]=alpha
    classifier['d'][iter,:]=d
    classifier['p'][iter]=p
    classifier['theta'][iter]=theta

# Show plots of training error and test error

train_errs = evaluateClassifier(classifier,X_train,t_train)
test_errs = evaluateClassifier(classifier,X_test,t_test)

plt.figure(1, figsize=(8, 5))
plt.plot(train_errs, 'r-', linewidth=2, label='Training error')
plt.plot(test_errs, 'b-', linewidth=2, label='Test error')
plt.xlabel('Number of iterations', fontsize=14)
plt.ylabel('Error rate', fontsize=14)
plt.title('AdaBoost Training vs Test Error', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


visualizeClassifier(classifier,2,(nx,ny))
