import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

def knn(mtx,n,metric='cosine'):
    dist = squareform(pdist(mtx,metric=metric))

    ranks = np.zeros((mtx.shape[0],mtx.shape[0]),dtype=int)
    for i in range(mtx.shape[0]):
        ranks[i][np.argsort(dist[i])] = np.arange(dist.shape[1])
    boolean = ranks < (n+1)
    boolean[np.identity(boolean.shape[0],dtype=bool)] = False
    return boolean


def sub_knn(mtx,sub=.5,n=10,intercon=10,metric='cosine'):
    intercon = 10
    connectivity = np.zeros((intercon,mtx.shape[0],mtx.shape[0]),dtype=bool)
    for i in range(intercon):
        mask = np.random.random(mtx.shape[0]) < sub
        double_mask = np.outer(mask,mask)
        sub_mtx = mtx[mask]
        sub_connectivity = knn(sub_mtx,n,metric=metric)
        connectivity[i][double_mask] = sub_connectivity.flatten()

    for i in range(0,1000000,11):
        segment_x = i%connectivity.shape[1]
        segment_y = (int(i/connectivity.shape[1]))%connectivity.shape[2]
        np.random.shuffle(connectivity[:,segment_x:segment_x+37,segment_y:segment_y+37])

    return connectivity


def fit_transform(mtx,cycles=10,sub=.3,n=10,metric='cosine',intercon=10,coordinates=None,no_plot=False):
    connectivity = sub_knn(mtx,sub=.5,intercon=intercon,n=n,metric=metric)
    final_index = -1 * np.ones(mtx.shape[0],dtype=int)
    density_estimate = np.ones(mtx.shape[0])

    running_connectivity = np.identity(counts.shape[0])
    for i in range(cycles):
        print(f"Estimating density:{i}")
        running_connectivity = np.dot(running_connectivity,skc[i%connectivity.shape[0]])

    density = np.sum(running_connectivity,axis=0)

    if not no_plot:
        if coordinates is None:
            tc = TSNE().fit_transform(mtx)
        else:
            tc = coordinates

    plt.figure()
    plt.title("Density Estimate")
    plt.scatter(tc[:,0],tc[:,1],s=1,c=np.log(density + 1))
    plt.show()

    density_ranked_samples = np.argsort(density)

    fully_connected = knn(mtx,n,metric=metric)
    fully_connected = np.logical_or(fully_connected,np.any(connectivity,axis=0))

    def ascend(index,connectivity,density,cache):
        ar = np.arange(connectivity.shape[1])
        current_index = index
        history = []
        for i in range(1000):
            if cache[current_index] > 0:
                current_index = cache[current_index]
                break
            neighbors = ar[connectivity[current_index]]
            neighbor_densities = density[neighbors]
            max_ndi = np.argmax(neighbor_densities)
            own_density = density[current_index]
            if own_density < neighbor_densities[max_ndi]:
                history.append(current_index)
                current_index = neighbors[max_ndi]
            else:
                break
        return current_index,history

    for i,sample in enumerate(density_ranked_samples):
        if i%100 == 0:
            print(f"Routing sample: {i}")
        destination,history = ascend(sample,fully_connected,density,final_index)
        final_index[sample] = destination
        final_index[np.array(history,dtype=int)] = destination


    print("Final clusters:")
    for final_cluster in set(final_index):
        print(final_cluster,np.sum(final_index == final_cluster))

    re_indexed = np.zeros(final_index.shape)
    for i,c in enumerate(set(final_index)):
        re_indexed[final_index == c] = i

    if not no_plot:
        plt.figure()
        plt.title("Clustering Visualization")
        plt.scatter(tc[:,0],tc[:,1],s=1,c=re_indexed,cmap='rainbow')
        plt.show()

    return re_indexed
