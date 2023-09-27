from model import KMeans
from utils import get_image, show_image, save_image, error
import numpy as np
from matplotlib import pyplot as plt


def main():
    # get image
    image = get_image('image.jpg')
    original_image = np.copy(image)
    img_shape = image.shape

    # reshape image
    image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    different_ks = []
    different_mse = []
    for num_clusters in [2, 5, 10, 20, 50]:
    # num_clusters = 20
        image = np.copy(original_image)
        image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
        kmeans = KMeans(num_clusters)

        # fit model
        kmeans.fit(image)

        # replace each pixel with its closest cluster center
        new_image = kmeans.replace_with_cluster_centers(image)

        # reshape image
        image_clustered = new_image.reshape(img_shape)
        current_mse = error(original_image, image_clustered)
        # Print the error
        print('MSE:', current_mse)
        different_ks.append(num_clusters)
        different_mse.append(current_mse)
        # show/save image
        # show_image(image)
        save_image(image_clustered, f'image_clustered_{num_clusters}.jpg')
        
    # save_image(original_image, f'image_original_{num_clusters}.jpg')
    plt.plot(different_ks, different_mse)
    plt.xlabel('Different #clusters values')
    plt.ylabel('MSE values')
    plt.title('Different #clusters values vs. MSE ')
    plt.savefig("Clusters_MSE.jpg")
    plt.clf()


if __name__ == '__main__':
    main()
