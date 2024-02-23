import numpy as np
from copy import deepcopy
from PIL import Image
uniform_background_color = [99, 133, 150]
black_color = [0, 0, 0]


specified_colors = {
    'Black': [113, 63, 29],
    'Latino': [174, 139, 97],
    'Arab': [219, 177, 128],
    'white': [234, 217, 217],
    'Zombie': [125, 162, 105],
    'Ape': [133, 111, 86],
    'Alien': [200, 251, 251]
}

def extract_id(url):
    return url.split('/')[-1].split('.')[0].split('cryptopunk')[-1]

def get_most_common_color(image):
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Reshape the array to a 2D array of pixels and 3 color values (RGB)
    pixels = image_array.reshape(-1, image_array.shape[-1])
    color_counts = {tuple(color): 0 for color in specified_colors}
    # Count occurrences of each specified color
    for label, color in specified_colors.items():
        color_array = np.array(color)
        mask = np.all(pixels == color_array, axis=1)
        color_counts[label] = np.sum(mask)

    # Find the most frequent specified color
    most_frequent_specified_label = max(color_counts, key=color_counts.get)
    most_frequent_specified_count = color_counts[most_frequent_specified_label]
    return most_frequent_specified_label
    # # Convert back to an image
    # modified_image = Image.fromarray(image_new)    
    # unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    # most_frequent_color = unique_colors[np.argmax(counts)]
    # image_array[(image_array == most_frequent_color).all(axis=-1)] = uniform_background_color
    # unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    # # Create masks to filter out the replacement color and black
    # is_not_replacement_color = ~np.all(unique_colors == uniform_background_color, axis=1)
    # is_not_black = ~np.all(unique_colors == black_color, axis=1)
    # is_not_excluded_color = is_not_replacement_color & is_not_black

    # # Filter out the blue background and black [0,0,0] borders
    # filtered_colors = unique_colors[is_not_excluded_color]
    # filtered_counts = counts[is_not_replacement_color]
    # skin_color = filtered_colors[np.argmax(filtered_counts)]
    # return skin_color