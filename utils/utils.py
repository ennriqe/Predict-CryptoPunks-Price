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

import matplotlib.pyplot as plt
import requests
import statsmodels.api as sm
import pandas as pd
import numpy as np

# Assuming X_test, y_test, and y_pred_gb_lgb are already defined

def plot_scatter_predictions_and_actuals_with_OLS(ax, X_test_subset, title):
    # Create the scatter plot
    ax.scatter(X_test_subset['y_test'], X_test_subset['y_pred_gb_lgb'])

    # Compute the OLS regression line without a constant
    X = X_test_subset['y_test']
    model = sm.OLS(X_test_subset['y_pred_gb_lgb'], X)
    results = model.fit()
    slope = results.params[0]

    # Plot the OLS regression line
    x_vals = np.array(ax.get_xlim())
    y_vals = slope * x_vals
    ax.plot(x_vals, y_vals, 'b-', label='OLS Line')

    # Determine the upper limit for the 45-degree line
    upper_limit = max(ax.get_xlim()[1], ax.get_ylim()[1])

    # Plot the 45-degree line starting from (0, 0)
    ax.plot([0, upper_limit], [0, upper_limit], 'r--', alpha=0.75, label='45-degree line')

    # Set the axes to start from (0, 0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Model's Prediction")
    ax.set_xlabel("Actual Value")
    # Set title and show the legend
    ax.set_title(title)
    ax.legend()
    
def exclude_top_and_bottom_5_percent(series):
    # Exclude the lowest 5%
    lower_threshold = np.percentile(series, 5)
    filtered_series = series[series > lower_threshold]
    
    # Exclude the top 5%
    upper_threshold = np.percentile(filtered_series, 95)
    final_filtered_series = filtered_series[filtered_series < upper_threshold]
    
    return final_filtered_series
def plot_eth_and_punk_prices(sales):
    sales['block_date'] = pd.to_datetime(sales.block_date)
    sales_sorted = sales.sort_values(by='block_date')
    mean_amount = sales_sorted.groupby('block_date').amount_original.mean()

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot on the first subplot
    axs[0].plot(np.log(mean_amount), label='Mean Punk Price in ETH')
    axs[0].plot(np.log(sales_sorted.groupby('block_date').Ethereum_Price.mean()), label='ETH Price')
    axs[0].plot(np.log(sales_sorted.groupby('block_date')['CP Price'].mean()), label='Punk Floor in ETH')
    axs[0].legend()

    axs[0].set_title('Log Prices')

    # Plot on the second subplot
    axs[1].plot(mean_amount)
    axs[1].set_title('Mean Punk Price in ETH')

    plt.show()

def get_ethereum_price_history():
    # URL of the API endpoint (using CoinGecko as an example)
    url = 'https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=max'

    # Make a request to the API
    response = requests.get(url)
    data = response.json()

    prices = data['prices']  # This might vary based on the API structure
    dates = [pd.to_datetime(price[0], unit='ms') for price in prices]
    eth_prices = [price[1] for price in prices]
    eth = pd.DataFrame({'Date': dates, 'Ethereum_Price': eth_prices})

    return eth