import numpy as np
import matplotlib.pyplot as plt
import requests
import statsmodels.api as sm
import pandas as pd


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
    """
    Extracts the id of the punk from the url

    Args:
    url (str): The url of the punk image

    Returns:
    str: The id of the punk
    """
    return url.split('/')[-1].split('.')[0].split('cryptopunk')[-1]

def get_most_common_color(image):
    """
    Returns the skin color of the cryptopunk

    Args:
    image (PIL.Image): The image to process

    Returns:
    str: The skin label of the punk
    """
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
    return most_frequent_specified_label

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
    
def exclude_top_and_bottom_x_percent(series, pecent):
    # Exclude the lowest 5%
    lower_threshold = np.percentile(series, pecent)
    filtered_series = series[series > lower_threshold]
    
    # Exclude the top 5%
    upper_threshold = np.percentile(filtered_series, 100 - pecent)
    final_filtered_series = filtered_series[filtered_series < upper_threshold]
    
    return final_filtered_series
def plot_eth_and_punk_prices(sales):
    """
    Plots the mean punk price, Ethereum price, and the punk floor in ETH

    Args:
    sales (pd.DataFrame): The sales data
    """
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
    """
    Fetches the historical Ethereum price data
    
    Returns:
    pd.DataFrame: Ethereum price history
    """
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

import requests
def get_punk_floor_today():
    """
    Fetches the current floor price of CryptoPunks using the OpenSea API

    Returns:
    float: The floor price of CryptoPunks in ETH
    """

    url = "https://api.opensea.io/api/v2/collections/cryptopunks/stats"

    headers = {
        "accept": "application/json",
        "x-api-key": "d2162577acee4cdfa2cf20e92f37409e"
    }

    response = requests.get(url, headers=headers)

    return response.json()['intervals'][1]['average_price']