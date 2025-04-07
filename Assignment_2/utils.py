import numpy as np

eps = np.finfo(float).eps

def sigmoid(a):
    # Compute logistic sigmoid
    #
    # s = sidmoid(a)
    s = 1/(1+np.exp(-a))
    return s

def plotMB(plt, w, w_old):
    # Plot the current separator in slop-intercept space (m-b) and draw a line between them

    m = -w[0]/w[1]
    b = -w[2]/w[1]

    m_old = -w_old[0]/(w_old[1]+eps)
    b_old = -w_old[2]/(w_old[1]+eps)
    plt.plot([m_old,m],[b_old,b],'ko-')

def drawSep(plt, w):
    # Draw a hyperaline w'x
    # w is [w_1 w_2 bias]'
    L = 1000

    # Check for degeneracy, at leat one of w[0] and w[1] must be large enough to invert
    if np.abs(w[1]) > eps:
        x1 = np.array([-L,L]).reshape(2,1)
        x2 = (-np.array([w[2],w[2]]) - w[0]*x1)/w[1]
    elif np.abs(w[0]) > eps:
        x2 = np.array([-L,L]).reshape(2,1)
        x1 = (-np.array([w[2],w[2]]) - w[2]*x2)/w[0]
    else:
        sys.exit('Invalid separator')

    plt.plot(x1,x2,'r-')

