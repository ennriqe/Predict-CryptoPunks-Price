import matplotlib.pyplot as plt
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
