import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from smooth_density_graph_rs import fit_predict as rust_fit_predict_reexport

from scipy.spatial.distance import pdist, squareform

def knn(mtx,k,metric='cosine',precomputed=None):
    from sklearn.neighbors import NearestNeighbors
    if precomputed is None:
        dist = squareform(pdist(mtx,metric=metric))
    else:
        dist = precomputed

    nbrs = NearestNeighbors(n_neighbors=k,metric='precomputed', algorithm='auto').fit(dist)
    return nbrs.kneighbors_graph().toarray()
    
    # ranks = np.zeros((mtx.shape[0],mtx.shape[0]),dtype=int)
    # for i in range(mtx.shape[0]):
    #     ranks[i][np.argsort(dist[i])] = np.arange(dist.shape[1])
    # boolean = ranks < (k+1)
    # boolean[np.identity(boolean.shape[0],dtype=bool)] = False
    # return boolean


def sub_knn(mtx,sub=.5,k=10,intercon=10,metric='cosine',precomputed=None):
    intercon = 10
    connectivity = np.zeros((intercon,mtx.shape[0],mtx.shape[0]),dtype=bool)
    for i in range(intercon):
        mask = np.random.random(mtx.shape[0]) < sub
        double_mask = np.outer(mask,mask)
        sub_mtx = mtx[mask]
        if precomputed is not None:
            sub_connectivity = knn(sub_mtx,k,metric=metric,precomputed=precomputed[mask].T[mask].T)
        else:
            sub_connectivity = knn(sub_mtx,k,metric=metric)
        connectivity[i][double_mask] = sub_connectivity.flatten()

    for i in range(0,1000000,11):
        segment_x = i%connectivity.shape[1]
        segment_y = (int(i/connectivity.shape[1]))%connectivity.shape[2]
        np.random.shuffle(connectivity[:,segment_x:segment_x+37,segment_y:segment_y+37])

    return connectivity


def fit_predict(mtx,cycles=10,sub=.3,k=10,metric='cosine',precomputed=None,intercon=10,coordinates=None,no_plot=False,rust=False,**kwargs):

    if rust:
        return rust_fit_predict(mtx,no_plot=no_plot,precomputed=bool(precomputed),k=k,**kwargs)

    if precomputed is None:
        distance_mtx = squareform(pdist(mtx,metric=metric))
    else:
        distance_mtx = precomputed
    connectivity = sub_knn(mtx,sub=sub,intercon=intercon,k=k,metric=metric,precomputed=distance_mtx)
    final_index = -1 * np.ones(mtx.shape[0],dtype=int)
    density_estimate = np.ones(mtx.shape[0])

    running_connectivity = np.identity(mtx.shape[0])
    for i in range(cycles):
        print(f"Estimating density:{100*(float(i+1)/cycles)}%",end='\r')
        running_connectivity = np.dot(running_connectivity,connectivity[i%connectivity.shape[0]])
    print("")

    density = np.sum(running_connectivity,axis=0)

    if not no_plot:
        if coordinates is None:
            tc = TSNE(metric='precomputed').fit_transform(distance_mtx)
        else:
            tc = coordinates

        plt.figure()
        plt.title("Density Estimate")
        plt.scatter(tc[:,0],tc[:,1],s=1,c=np.log(density + 1))
        plt.show()

    density_ranked_samples = np.argsort(density)

    fully_connected = knn(mtx,k,metric=metric,precomputed=distance_mtx)
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
            print(f"Routing samples: {100*(float(i+1)/len(density_ranked_samples))}%", end='\r')
        destination,history = ascend(sample,fully_connected,density,final_index)
        final_index[sample] = destination
        final_index[np.array(history,dtype=int)] = destination
    print("")

    print("Final clusters:")
    final_clusters = list(set(final_index))
    final_cluster_sizes = [np.sum(final_index == cluster) for cluster in final_clusters]
    final_clusters_sorted = list(reversed(list(np.array(final_clusters)[np.argsort(final_cluster_sizes)])))

    plt.figure()
    plt.title("Distribution of Cluster Sizes")
    plt.bar(np.arange(len(final_clusters)),list(reversed(sorted(final_cluster_sizes))))
    plt.show()

    # for final_cluster in set(final_index):
    #     print(final_cluster,np.sum(final_index == final_cluster))
    print(f"Total clusters:{len(set(final_index))}")

    re_indexed = np.zeros(final_index.shape)
    for i,c in enumerate(final_clusters_sorted):
        re_indexed[final_index == c] = i

    if not no_plot:
        plt.figure()
        plt.title("Clustering Visualization")
        plt.scatter(tc[:,0],tc[:,1],s=1,c=re_indexed,cmap='rainbow')
        plt.show()

    return re_indexed

# def fit_predict(targets,command="fitpredict",auto=True,no_plot=False,precomputed=False,verbose=False,backtrace=False,**kwargs):
#
#     labels =  rust_fit_predict_reexport(targets,command="fitpredict",auto=True,precomputed=False,verbose=False,backtrace=False,**kwargs)
#
#     if not no_plot:
#         tc = TSNE().fit_transform(mtx)
#
#         plt.figure()
#         plt.title("Density Estimate")
#         plt.scatter(tc[:,0],tc[:,1],s=1,c=labels )
#         plt.show()
#
#     return labels

def rust_fit_predict(targets,command="fitpredict",auto=True,no_plot=False,precomputed=False,verbose=False,backtrace=False,**kwargs):

    labels =  rust_fit_predict_reexport(targets,command="fitpredict",auto=True,precomputed=False,verbose=False,backtrace=False,**kwargs)

    if not no_plot:
        tc = TSNE().fit_transform(mtx)

        plt.figure()
        plt.title("Density Estimate")
        plt.scatter(tc[:,0],tc[:,1],s=1,c=labels )
        plt.show()

    return labels
