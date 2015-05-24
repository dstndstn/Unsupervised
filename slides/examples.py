import numpy as np
import pylab as plt

from astrometry.util.plotutils import *

def example1d():
    means = [ 4  , 8   ]
    stds  = [ 1.5, 1   ]
    amps  = [ 0.7, 0.3 ]

    K = len(means)
    N = 300
    amps = np.array(amps)
    amps /= np.sum(amps)
    nk = np.random.multinomial(N, amps)
    print 'nk', nk
    x = []
    for n,mean,std in zip(nk, means, stds):
        x.extend(np.random.normal(loc=mean, scale=std, size=n))

    plt.clf()
    plt.hist(x, histtype='step', range=(0,12), bins=25)
    plt.savefig('ex1.png')    


def example2d():
    means = [ (4,3.5)  , (8,4.5) ]
    stds  = [ 1.5, 1   ]
    amps  = [ 0.7, 0.3 ]

    K = len(means)
    D = 2
    N = 300
    amps = np.array(amps)
    amps /= np.sum(amps)
    nk = np.random.multinomial(N, amps)
    print 'nk', nk
    x = []
    for n,mean,std in zip(nk, means, stds):
        x.append(np.random.normal(loc=mean, scale=std, size=(n,D)))

    print [xi.shape for xi in x]

    ax = [-1, 11, -1, 9]
    for colors,plotname in [('bb','a'), ('br','b')]:
        plt.clf()
        for xi,cc in zip(x, colors):
            plt.plot(xi[:,0], xi[:,1], 'o', color=cc, mec='none', alpha=0.8)
            plt.plot(xi[:,0], xi[:,1], 'o', color='k', mfc='none', alpha=0.5)
        plt.axis(ax)
        plt.xticks([]); plt.yticks([])
        plt.savefig('ex2%s.pdf' % plotname)


def sample_clusters(amps, means, stds, N=100, D=2):
    K = len(amps)
    amps = np.array(amps)
    amps /= np.sum(amps)
    nk = np.random.multinomial(N, amps)
    x = []
    for n,mean,std in zip(nk, means, stds):
        x.append(np.random.normal(loc=mean, scale=std, size=(n,D)))
    print [xi.shape for xi in x]
    return x

def get_clusterA():
    means = [ (3.5,2.5)  , (7.5,3.5), (4.5,6.5) ]
    stds  = [ 1.,  0.7, 0.7   ]
    amps  = [ 0.5, 0.25, 0.25 ]
    return amps, means, stds

def example3():
    C = get_clusterA()
    x = sample_clusters(*C, N=200)
    
    ax = [-1, 11, -1, 9]
    for colors,plotname in [('kkk','a'), ('brg','b')]:
        plt.clf()
        for xi,cc in zip(x, colors):
            plt.plot(xi[:,0], xi[:,1], 'o', color=cc, mec='none', alpha=0.8)
            #plt.plot(xi[:,0], xi[:,1], 'o', color=cc, mec='none')
            plt.plot(xi[:,0], xi[:,1], 'o', mec='k', mfc='none', alpha=0.1)

        # import matplotlib as mpl
        # for m,s,c in zip(means, stds, colors):
        #     plt.gca().add_patch(mpl.patches.Circle(
        #         m, radius=s, ec=c, fc='none', zorder=20, lw=3))
        plt.axis(ax)
        plt.xticks([]); plt.yticks([])
        plt.xlabel('Measurement A')
        plt.ylabel('Measurement B')
        #plt.axis('equal')
        plt.savefig('ex3%s.pdf' % plotname)

def distance_matrix_2d(A, B):
    Na,D = A.shape
    Nb,Db = B.shape
    assert(Db == D)
    dists = np.zeros((Na,Nb))
    for a in range(Na):
        dists[a,:] = np.sqrt(np.sum((A[a] - B)**2, axis=1))
    return dists

def voronoi_plot_2d(vor, ax=None):
    ptp_bound = vor.points.ptp(axis=0)

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


def kmeans():
    C = get_clusterA()
    N = 200
    #N = 8
    X = sample_clusters(*C, N=N)
    X = np.vstack(X)
    print X.shape

    ps = PlotSequence('kmeans', suffix='pdf')
    
    K = 3
    centroids = X[np.random.permutation(N)[:K],:]
    print 'centroids', centroids

    ax = [0, 10, 0, 8.5]

    while True:
        # compute nearest centroid for data points
        dists = distance_matrix_2d(centroids, X)
        print 'dists', dists.shape
        #print 'dists', dists
        nearest = np.argmin(dists, axis=0)
        print 'nearest', nearest.shape, np.unique(nearest)
        #print nearest
        
        colors = 'brgmck'
        plt.clf()
        for i,c in enumerate(centroids):
            plt.plot(c[0], c[1], 'x', mew=2, ms=15, color=colors[i])
        for i,c in enumerate(centroids):
            I = np.flatnonzero(nearest == i)
            plt.plot(X[I,0], X[I,1], '.', mfc=colors[i], mec='k')

        from scipy.spatial import Voronoi #, voronoi_plot_2d
        vor = Voronoi(centroids)
        voronoi_plot_2d(vor, plt.gca())
        plt.axis(ax)
        plt.xticks([]); plt.yticks([])
        ps.savefig()

        newcentroids = []
        for i,c in enumerate(centroids):
            I = np.flatnonzero(nearest == i)
            newcentroids.append((np.mean(X[I,:], axis=0)))

        plt.clf()
        for i,(oc,nc) in enumerate(zip(centroids, newcentroids)):
            plt.plot(oc[0], oc[1], 'kx', mew=2, ms=10)
            plt.arrow(oc[0], oc[1],nc[0]-oc[0], nc[1]-oc[1], color=colors[i])
            plt.plot(nc[0], nc[1], 'x', mew=2, ms=15, color=colors[i])
        for i,c in enumerate(centroids):
            I = np.flatnonzero(nearest == i)
            plt.plot(X[I,0], X[I,1], '.', mfc=colors[i], mec='k')
        voronoi_plot_2d(vor, plt.gca())
        plt.axis(ax)
        plt.xticks([]); plt.yticks([])
        ps.savefig()

        newcentroids = np.array(newcentroids)

        print 'Centroid difference:', newcentroids - centroids
        
        if np.max(np.abs(centroids - newcentroids)) < 1e-8:
            break
        
        centroids = newcentroids


plt.figure(figsize=(4,3))
plt.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.95)
    
np.random.seed(42)    
#example1d()
#example2d()
#example3()
#np.random.seed(42)    
kmeans()   
