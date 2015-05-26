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
    trueclass = []
    for n,mean,var in zip(nk, means, varis):
        xi = np.random.normal(loc=mean, scale=np.sqrt(var), size=(n,D))
        if bounds is not None:
            outofbounds = np.empty(len(xi), bool)
            while True:
                outofbounds[:] = False
                for d,(lo,hi) in enumerate(bounds):
                    outofbounds |= np.logical_or(xi[:,d] < lo, xi[:,d] > hi)
                if not np.any(outofbounds):
                    break
                xi[outofbounds,:] = np.random.normal(
                    loc=mean, scale=std, size=(np.sum(outofbounds),D))
            
        x.append(xi)
    # print [xi.shape for xi in x]
    return x

def get_clusters_A():
    '''
    Returns parameters of an example isotropic cluster, for K-means demo.
    '''
    amps  = [ 0.5, 0.25, 0.25 ]
    means = [ (3.5,2.5)  , (7.5,3.5), (4.5,6.5) ]
    varis  = [ 1.,  0.7, 0.7   ]
    ax = [0, 10, 0, 8.5]
    return (amps, means, varis), ax

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

        
