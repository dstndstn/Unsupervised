import numpy as np

colors = 'brgmck'

def sample_clusters(amps, means, varis, N=100, D=2, bounds=None):
    '''
    Draws samples from a set of clusters.

    *amps*: [iterable] cluster weights; fraction of points drawn from each
    *means*: [iterable of D-vectors] cluster centers, eg
        [ [0,0], [1,2] ] describes one cluster at the origin and one at [1,2].
    *varis*: [iterable] variances of the clusters
    *N*: number of samples to draw
    *D*: dimensionality of the samples
    *bounds*: [iterable of (lo,hi) pairs] -- re-draw any samples outside this box

    Returns a list of arrays with shape (n,D), where each array is
    drawn from one cluster.
    '''
    K = len(amps)
    amps = np.array(amps)
    amps /= np.sum(amps)
    nk = np.random.multinomial(N, amps)
    x = []

    def _sample_gaussian(mean, var, N, D):
        if np.isscalar(var):
            # Assume isotropic
            return np.random.normal(loc=mean, scale=np.sqrt(var), size=(N,D))
        # Assume covariance matrix
        var = np.array(var)
        u,s,v = np.linalg.svd(var)
        x = np.random.normal(size=(N,D))
        return mean + np.dot(u, (x * s).T).T
        
    for n,mean,var in zip(nk, means, varis):
        xi = _sample_gaussian(mean, var, n, D)
        if bounds is not None:
            outofbounds = np.empty(len(xi), bool)
            while True:
                outofbounds[:] = False
                for d,(lo,hi) in enumerate(bounds):
                    outofbounds |= np.logical_or(xi[:,d] < lo, xi[:,d] > hi)
                if not np.any(outofbounds):
                    break
                xi[outofbounds,:] = _sample_gaussian(mean, var,
                                                     np.sum(outofbounds), D)
            
        x.append(xi)
    # print [xi.shape for xi in x]
    return x

def get_clusters_A():
    '''
    Returns parameters of an example isotropic cluster, for K-means demo.
    '''
    amps  = [ 0.5, 0.25, 0.25 ]
    means = [ (3.5,2.5)  , (7.5,3.5), (4.5,6.5) ]
    varis  = [ 1.**2,  0.7**2, 0.7**2 ]
    ax = [0, 10, 0, 8.5]
    return (amps, means, varis), ax

def get_clusters_C():
    '''
    Returns parameters of an example isotropic cluster, for K-means demo.
    This one has 90% of the mass in one component and tends to mess up K-means.
    '''
    amps  = [ 0.9, 0.1 ]
    means = [ (3.5, 4.), (6.5, 4.) ]
    varis  = [ 0.8**2, 0.5**2 ]
    ax = [0, 10, 0, 8]
    return (amps, means, varis), ax

def get_clusters_D():
    '''
    Returns parameters of a 2-D general Gaussian mixture model
    '''
    amps = [0.8, 0.2]
    means = [ (3., 4.), (6.5, 4.) ]
    covs = [ np.array([[1.,-0.5],[-0.5,1.]]),
             np.array([[1.,0.5],[0.5,1.]]),
             ]
    ax = [0, 10, 0, 8]
    return (amps,means,covs), ax


def distance_matrix(A, B):
    '''
    Given two sets of data points, computes the Euclidean distances
    between each pair of points.

    *A*: (N, D) array of data points
    *B*: (M, D) array of data points

    Returns: (N, M) array of Euclidean distances between points.
    '''
    Na,D = A.shape
    Nb,Db = B.shape
    assert(Db == D)
    dists = np.zeros((Na,Nb))
    for a in range(Na):
        dists[a,:] = np.sqrt(np.sum((A[a] - B)**2, axis=1))
    return dists

# Copied and very slightly modified from scipy
def voronoi_plot_2d(vor, ax=None):
    #ptp_bound = vor.points.ptp(axis=0)
    ptp_bound = np.array([1000,1000])
    
    center = vor.points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            ax.plot([vor.vertices[i,0], far_point[0]],
                    [vor.vertices[i,1], far_point[1]], 'k--')

def plot_kmeans(i, X, K, centroids, newcentroids, nearest, show=True):
    import pylab as plt
    plt.clf()
    plotsymbol = 'o'
    if nearest is None:
        distances = distance_matrix(X, centroids)
        nearest = np.argmin(distances, axis=1)
        
    for i,c in enumerate(centroids):
        I = np.flatnonzero(nearest == i)
        plt.plot(X[I,0], X[I,1], plotsymbol, mfc=colors[i], mec='k')
    ax = plt.axis()
    for i,(oc,nc) in enumerate(zip(centroids, newcentroids)):
        plt.plot(oc[0], oc[1], 'kx', mew=2, ms=10)
        plt.plot([oc[0], nc[0]], [oc[1], nc[1]], '-', color=colors[i])
        plt.plot(nc[0], nc[1], 'x', mew=2, ms=15, color=colors[i])
        
    vor = None
    if K > 2:
        from scipy.spatial import Voronoi #, voronoi_plot_2d
        vor = Voronoi(centroids)
        voronoi_plot_2d(vor, plt.gca())
    else:
        mid = np.mean(centroids, axis=0)
        x0,y0 = centroids[0]
        x1,y1 = centroids[1]
        slope = (y1-y0)/(x1-x0)
        slope = -1./slope
        run = 1000.
        plt.plot([mid[0] - run, mid[0] + run],
                 [mid[1] - run*slope, mid[1] + run*slope], 'k--')
    plt.axis(ax)
    if show:
        plt.show()

        
def gaussian_probability(X, mean, cov):
    '''
    Returns the probability of drawing data points from a Gaussian distribution

    *X*: (N,D) array of data points
    *mean*: (D,) vector: mean of the Gaussian
    *cov*: (D,D) array: covariance of the Gaussian

    Returns: (N,) vector of Gaussian probabilities
    '''
    D,d = cov.shape
    assert(D == d)

    # I haven't found a beautiful way of writing this in numpy...
    mahal = np.sum(np.dot(np.linalg.inv(cov), (X - mean).T).T * (X - mean),
                   axis=1)
    return (1./((2.*np.pi)**(D/2.) * np.sqrt(np.linalg.det(cov)))
            * np.exp(-0.5 * mahal))

def plot_ellipse(mean, cov, *args, **kwargs):
    import pylab as plt
    u,s,v = np.linalg.svd(cov)
    angle = np.linspace(0., 2.*np.pi, 200)
    u1 = u[0,:]
    u2 = u[1,:]
    s1,s2 = np.sqrt(s)
    xy = (u1[np.newaxis,:] * s1 * np.cos(angle)[:,np.newaxis] +
          u2[np.newaxis,:] * s2 * np.sin(angle)[:,np.newaxis])
    return plt.plot(mean[0] + xy[:,0], mean[1] + xy[:,1], *args, **kwargs)
    
    
def plot_em(step, X, K, amps, means, covs, z,
            newamps, newmeans, newcovs, show=True):
    import pylab as plt
    from matplotlib.colors import ColorConverter

    (N,D) = X.shape

    if z is None:
        z = np.zeros((N,K))
        for k,(amp,mean,cov) in enumerate(zip(amps, means, covs)):
            z[:,k] = amp * gaussian_probability(X, mean, cov)
        z /= np.sum(z, axis=1)[:,np.newaxis]
    
    plt.clf()
    # snazzy color coding
    cc = np.zeros((N,3))
    CC = ColorConverter()
    for k in range(K):
        rgb = np.array(CC.to_rgb(colors[k]))
        cc += z[:,k][:,np.newaxis] * rgb[np.newaxis,:]

    plt.scatter(X[:,0], X[:,1], color=cc, s=9, alpha=0.5)

    ax = plt.axis()
    for k,(amp,mean,cov) in enumerate(zip(amps, means, covs)):
        plot_ellipse(mean, cov, 'k-', lw=4)
        plot_ellipse(mean, cov, 'k-', color=colors[k], lw=2)

    plt.axis(ax)
    if show:
        plt.show()

def plot_gmm_samples(X, K, params):
    import pylab as plt
    nwalkers,ndim = params.shape
    plt.clf()
    plt.scatter(X[:,0], X[:,1], color='k', s=9, alpha=0.5)
    N,D = X.shape
    for i in range(nwalkers):
        logamps,means,covs = unpack_gmm_params(params[i,:], K, D)
        amps = np.exp(np.append(1, logamps))
        amps /= np.sum(amps)
        for k,(amp,mean,cov) in enumerate(zip(amps, means, covs)):
            plot_ellipse(mean, cov, '-', color=colors[k], lw=1, alpha=0.2)
        
def unpack_gmm_params(params, K, D):
    amps   = params[:K-1]
    params = params[K-1:]
    means  = params[:K*D].reshape((K,D))
    params = params[K*D:]
    covs = np.zeros((K,D,D))
    # we have to unpack the covariances carefully (triangular matrix)
    tri = np.tri(D)
    I = np.flatnonzero(tri)
    for k in range(K):
        covs[k,:,:].flat[I] = params[:len(I)]
        params = params[len(I):]
        # copy lower triangle to upper triangle
        covs[k,:,:] += (covs[k,:,:].T * (1 - tri))
    return amps, means, covs

def pack_gmm_params(amps, means, covs):
    K = len(amps)
    k,D = means.shape
    assert(k == K)
    k,d1,d2 = covs.shape
    assert(k == K)
    assert(d1 == D)
    assert(d2 == D)
    pp = [amps[:-1], means.ravel()]
    # grab the lower triangular matrix elements;
    # 'tri' has ones in the lower diagonal
    tri = np.tri(D)
    # 'I' gives the flattened matrix elements in the lower diagonal
    I = np.flatnonzero(tri)
    for k in range(K):
        pp.append(covs[k,:,:].flat[I])
    return np.hstack(pp)

def gaussian_probability_1d(x, mean, vari):
    '''
    Returns the probability of drawing data points from a Gaussian distribution

    *X*: (N,) array of data points
    *mean*: scalar: mean of the Gaussian
    *vari*: scalar: variance of the Gaussian

    Returns: (N,) vector of Gaussian probabilities
    '''
    # I haven't found a beautiful way of writing this in numpy...
    mahal = (x - mean)**2 / vari
    return (1./np.sqrt(2.*np.pi * vari)
            * np.exp(-0.5 * mahal))

def plot_sinusoid_samples(Xi, xf, params):
    import pylab as plt
    nwalkers,ndim = params.shape
    plt.clf()
    for i,X in enumerate(Xi):
        plt.scatter(X[:,0], X[:,1], color=colors[i], s=9, alpha=0.5)
    for i in range(nwalkers):
        fg,offset,amp = params[i,:]
        ypred = offset + amp * np.sin(xf)
        plt.plot(xf, ypred, '-', color=colors[0], alpha=0.2)



if __name__ == '__main__':
    a = np.array([1,2,3])
    means = (1+np.arange(6)).reshape(3,2)
    covs = (100 + np.arange(12)).reshape(3,2,2)
    P = pack_gmm_params(a, means, covs)
    print 'Packed params:', P

    K = 3
    D = 2
    a2,m2,c2 = unpack_gmm_params(P, K, D)
    print 'a2', a2
    print 'm2', m2
    print 'c2', c2
    
