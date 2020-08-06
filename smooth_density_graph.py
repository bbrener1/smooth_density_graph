import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from smooth_density_graph_rs import fit_predict as rust_fit_predict_reexport

from scipy.spatial.distance import pdist, squareform



def sub_knn(mtx,sub=.5,k=10,intercon=10,metric='cosine',shuffle=3):

    connectivity = np.zeros((intercon,mtx.shape[0],mtx.shape[0]),dtype=bool)

    for i in range(intercon):
        print(f"Subsampling connectivity: {i}")
        mask = np.random.random(mtx.shape[0]) < sub
        double_mask = np.outer(mask,mask)
        sub_mtx = mtx[mask]
        sub_connectivity = dense_neighbors(fast_knn(sub_mtx,k,metric=metric))
        connectivity[i][double_mask] = sub_connectivity.flatten()

    print("Shuffling")

    shuffle_range = int((connectivity.shape[1] / 11) * connectivity.shape[1])

    print(f"Shuffle range: {shuffle_range}")

    for i in range(0,shuffle_range,11):
        segment_x = i%connectivity.shape[1]
        segment_y = (int(i/connectivity.shape[1]))%connectivity.shape[2]
        np.random.shuffle(connectivity[:,segment_x:segment_x+37,segment_y:segment_y+37])

    return connectivity


def fit_predict(mtx,cycles=10,sub=.3,k=10,metric='cosine',precomputed=None,intercon=10,coordinates=None,no_plot=False,rust=False,shuffle=3,**kwargs):

    if rust:
        return rust_fit_predict(mtx,no_plot=no_plot,precomputed=bool(precomputed),k=k,**kwargs)

    connectivity = sub_knn(mtx,sub=sub,intercon=intercon,k=k,metric=metric,shuffle=shuffle)
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



def rust_fit_predict(targets,command="fitpredict",auto=True,no_plot=False,precomputed=False,verbose=False,backtrace=False,**kwargs):

    labels =  rust_fit_predict_reexport(targets,command="fitpredict",auto=True,precomputed=False,verbose=False,backtrace=False,**kwargs)

    if not no_plot:
        tc = TSNE().fit_transform(mtx)

        plt.figure()
        plt.title("Density Estimate")
        plt.scatter(tc[:,0],tc[:,1],s=1,c=labels )
        plt.show()

    return labels

from scipy.spatial.distance import pdist,cdist,squareform

def fast_knn(elements, k, neighborhood_fraction=.01, metric='cosine'):

    nearest_neighbors = np.zeros((elements.shape[0], k), dtype=int)
    guarantee = np.zeros(elements.shape[0], dtype=bool)

    neighborhood_size = max(
        k * 3, int(elements.shape[0] * neighborhood_fraction))
    anchor_loops = 0

    while np.sum(guarantee) < guarantee.shape[0]:

        anchor_loops += 1

        available = np.arange(guarantee.shape[0])[~guarantee]
        np.random.shuffle(available)
        anchors = available[:int(guarantee.shape[0] / neighborhood_size) * 3]

        for anchor in anchors:
            print(f"Complete:{np.sum(guarantee)}\r", end='')

            anchor_distances = cdist(elements[anchor].reshape(
                1, -1), elements, metric=metric)[0]

            neighborhood = np.argpartition(anchor_distances, neighborhood_size)[
                :neighborhood_size]
            anchor_local = np.where(neighborhood == anchor)[0]

            local_distances = squareform(
                pdist(elements[neighborhood], metric=metric))
            local_distances[np.identity(
                local_distances.shape[0], dtype=bool)] = float('inf')

            anchor_distances = local_distances[anchor_local]

            for i, sample in enumerate(neighborhood):
                if not guarantee[sample]:

                    best_neighbors_local = np.argpartition(
                        local_distances[i], k)
                    best_neighbors = neighborhood[best_neighbors_local[:k]]

                    worst_best_local = best_neighbors_local[k]
                    worst_best_local_distance = local_distances[i,
                                                                worst_best_local]

                    worst_local = np.argmax(local_distances[i])
                    anchor_to_worst = local_distances[anchor_local,
                                                      worst_local]

                    anchor_distance = local_distances[anchor_local, i]

                    criterion_distance = anchor_to_worst - anchor_distance

                    if worst_best_local_distance <= criterion_distance:
                        continue
                    else:
                        nearest_neighbors[sample] = best_neighbors
                        guarantee[sample] = True
    print("\n")

    return nearest_neighbors

def dense_neighbors(sparse):

    dense = np.zeros((sparse.shape[0],sparse.shape[0]),dtype=bool)
    dense[(np.arange(sparse.shape[0]),sparse.T)] = True
    return dense
