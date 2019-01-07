import numpy as np
from scipy.stats.mstats import gmean
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist,squareform

def compute_neighbors(mtx,k,n_jobs=-1,metric='cosine'):

    return NearestNeighbors(n_neighbors=k,n_jobs=n_jobs,metric=metric).fit(mtx).kneighbors()

def neighborhood_models(mtx,sample_fraction=.1,neighbors=5,models=20,metric='cosine'):

    neighbor_models = []

    for i in range(models):

        subsample_mask = np.random.random(mtx.shape[0]) < sample_fraction

        subsample = mtx[subsample_mask]

        model = NearestNeighbors(n_neighbors=neighbors,metric=metric).fit(subsample)

        neighbor_models.append((model,subsample))

    return neighbor_models

def walk_the_models(point,models,metric='cosine'):

    steps = []
    current_point = point
    previous_point = point

    counter = 0

    # while True:
    # for _ in range(10001):
    for _ in range(100):

        model_index = counter % len(models)
        model,subsample = models[model_index]
        neighbor_indecies = model.kneighbors(current_point.reshape(1, -1),return_distance=False)[0]
        neighbors = subsample[neighbor_indecies]
        new_point = np.mean(neighbors,axis=0) * .3 + current_point * .77
        previous_point = current_point
        current_point = new_point
        shift = cdist([previous_point,],[current_point,],metric=metric)[0,0]
        if len(steps) >= len(models):
            steps[model_index] = shift
            # if steps[(counter + 1) % len(models)] <= shift:
            #     break
        else:
            steps.append(shift)
        counter += 1
        # print(counter)
        # print(previous_point)
        # print(current_point)
        # print(shift)
        if counter > 10000:
            print("Too many loops")
            break

    return current_point,np.mean(steps)

def fuzzy_walk(point,models,metric='cosine'):

    fuzzy_points = np.zeros((min(len(models),5),len(point)))

    for i in range(min(len(models),5)):
        permuted_models = models[i:]+models[:i]
        fuzzy_points[i],step_fuzz = walk_the_models(point,permuted_models)

    # print(fuzzy_points)
    final_point = np.mean(fuzzy_points,axis=0)
    fuzz = np.mean(cdist(fuzzy_points,[final_point],metric=metric))
    # print((final_point,fuzz))
    return (final_point,fuzz)


def converge(mtx,sample_fraction=.1,neighbors=5,models=20,metric='cosine'):

    nm = neighborhood_models(mtx,sample_fraction,neighbors,models,metric)

    final_positions = np.zeros(mtx.shape)
    fuzz = np.zeros(mtx.shape[0])

    for i,point in enumerate(mtx):
        print(i)
        final_position,point_fuzz = fuzzy_walk(point,nm,metric)
        # final_position,point_fuzz = walk_the_models(point,nm,metric)
        final_positions[i] = final_position
        fuzz[i] = point_fuzz

    return final_positions,fuzz

def quick_cluster(final_points,fuzz,metric='cosine'):

    clusters = {0:(final_points[0],0,[0,])}
    cluster_list = []

    for point in range(1,len(final_points)):
        print(point)
        print(f"Clusters:{len(clusters)}")
        distances = [(cluster,cdist([clusters[cluster][0],],[final_points[point],],metric=metric)[0,0]) for cluster in clusters]
        best_cluster,distance = min(distances,key=lambda x: x[1])
        print(f"best:{best_cluster},{distance}")
        print(f"r:{clusters[best_cluster][1]}")
        print(f"f:{fuzz[point]}")
        if distance < 2 * fuzz[point] + clusters[best_cluster][1]:
                center,radius,members = clusters[best_cluster]
                members.append(point)
                member_coordinates = final_points[members]
                center = cluster_center(member_coordinates)
                radius = cluster_radius(member_coordinates,center)
                clusters[best_cluster] = (center,radius,members)
                cluster_list.append(best_cluster)
        else:
            new_cluster = len(clusters)
            clusters[new_cluster] = (final_points[point],0,[point,])
            cluster_list.append(new_cluster)

    return cluster_list

# def merge_clusters(clusters):
#     for c1 in clusters:
#         for c2 in clusters:
#             if c1

def cluster_center(points):
    return np.mean(np.array(points),axis=0)

def cluster_radius(points,center):
    distances = cdist([center,],points)
    return np.mean(distances.flatten())
