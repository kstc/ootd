from sklearn.cluster import KMeans
from scipy.stats import itemfreq
from PIL import Image

import numpy as np
import operator


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return np.asarray(image, dtype=np.uint8)

def get_dominant_color(image, n_colors=3, show=False):

    input_image = image
    image_array = np.array(input_image, dtype=np.uint8) #Convert to numpy array of type uint8

    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(image_array.shape) # shape returns width, height, color space (rbg)
    assert d == 3 # make sure color space is rgb
    image_array = np.reshape(input_image, (w * h, d)) # w*h = number of pixels, d = rgb

    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)
    labels = kmeans.labels_

    new = recreate_image(kmeans.cluster_centers_, labels, w, h) # Converting the image to uint8 datatype
    
    if show:
        # Display quantized image
        im = Image.fromarray(new)
        im.show()
        im.save("out.bmp")

    # Get int conversions from center clusters
    clusters = np.uint8(kmeans.cluster_centers_)

    # Look for dominant
    r = new.shape[0] # Get size of y
    px1 = [p for p in new[0][0]] # Pixel value for topmost, leftmost pixel, indicates background
    px2 = [p for p in new[r - 1][0]] # Pixel value for bottommost, leftmost pixel, indicates background

    # Sort by cluster sizes, the larger the cluster, the more dominant 
    cluster_sizes  = dict(itemfreq(labels))
    sorted_cluster = sorted(cluster_sizes.items(), key=operator.itemgetter(1), reverse=True)

    # Loop through list of cluster indices, sorted by cluster size
    dominant = None
    for c in sorted_cluster:
        index = c[0]
        # If cluster values are not equal to the values in px1 and px2, it is not background! THIS IS IT
        if not (px1 == list(clusters[index]) or px2 == list(clusters[index])):
            dominant = list(clusters[index])
            break

    print "DOMINANT", dominant, "(RGB)"
    dominant = (int(dominant[0]) * 65536) + (int(dominant[1]) * 256) + int(dominant[2])
    return dominant

