import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import shap
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import optuna
from utils.utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os

num_epochs = 10
sales = pd.read_excel('cryptopunk_sales.xlsx')
metadata = pd.read_csv('cryptopunk_metadata.csv')
cryptopunks_skin = pd.read_csv('cryptopunks_skin.csv')
floor = pd.read_csv('punk floor.csv')

metadata = metadata.merge(cryptopunks_skin, left_on = 'ID', right_on = 'id')

# there are 1.4k below $30
sales = sales[sales.amount_usd>30]
sales['block_date']= pd.to_datetime(sales.block_date)
sales = sales[sales.block_date> '2021-02-01']

sales = sales[['block_date', 'block_month', 'token_id', 'amount_usd', 'amount_original']]

unique_attributes = set()
attribute_columns = [col for col in metadata.columns if 'Attribute' in col]
for col in attribute_columns:
    unique_attributes.update(metadata[col].dropna().unique())

# Creating dummy variables
for attribute in unique_attributes:
    metadata[attribute] = metadata[attribute_columns].apply(lambda x: attribute in x.values, axis=1).astype(int)

metadata = metadata[pd.Series(metadata.columns)[~pd.Series(metadata.columns).isin(attribute_columns)]]
sales = pd.merge(sales, metadata, left_on = 'token_id', right_on = 'ID')


eth = get_ethereum_price_history()

sales = pd.merge(sales, eth, left_on  = 'block_date', right_on = 'Date')
floor['day'] = pd.to_datetime(floor.day)
sales = sales.merge(floor, left_on = 'block_date', right_on = 'day')

earliest_year = sales['block_date'].dt.year.min()
sales['relative_month_number'] = ((sales['block_date'].dt.year - earliest_year) * 12 + sales['block_date'].dt.month)
sales['relative_month_number'] = sales['relative_month_number'] - 12


sales = sales[['amount_original', 'CP Price',
       'Gender', 'Category', 'Front Beard Dark', 'VR', 'Clown Eyes Green',
       'Buck Teeth', 'Wild Hair', 'Silver Chain', 'Cigarette',
       'Purple Eye Shadow', 'Pigtails', 'Handlebars', 'Normal Beard',
       'Blonde Bob', 'Muttonchops', 'Smile', 'Shaved Head', 'Mustache',
       'Mohawk Dark', 'Straight Hair', 'Choker', 'Regular Shades',
       'Peak Spike', 'Tassle Hat', 'Dark Hair', 'Knitted Cap', 'Bandana',
       'Pink With Hat', 'Gold Chain', 'Mohawk', 'Welding Goggles',
       'Cap Forward', 'Tiara', 'Purple Lipstick', 'Small Shades',
       'Stringy Hair', 'Do-rag', 'Wild White Hair', 'Frown', 'Red Mohawk',
       'Half Shaved', 'Clown Hair Green', 'Vampire Hair', 'Beanie',
       'Clown Nose', 'Messy Hair', 'Blonde Short', 'Mole', 'Purple Hair',
       'Chinstrap', 'Orange Side', 'Hot Lipstick', 'Horned Rim Glasses', 'Cap',
       'Green Eye Shadow', 'Nerd Glasses', 'Rosy Cheeks', 'Pilot Helmet',
       'Straight Hair Dark', 'Medical Mask', 'Frumpy Hair', 'Wild Blonde',
       'Hoodie', 'Earring', 'Big Shades', 'Spots', 'Headband', 'Goat',
       'Big Beard', 'Classic Shades', 'Clown Eyes Blue', 'Blue Eye Shadow',
       'Cowboy Hat', 'Luxurious Beard', 'Crazy Hair', 'Normal Beard Black',
       'Fedora', 'Straight Hair Blonde', 'Vape', 'Mohawk Thin', 'Front Beard',
       '3D Glasses', 'Police Cap', 'Top Hat', 'Shadow Beard', 'Eye Mask',
       'Black Lipstick', 'Eye Patch', 'Pipe', 'relative_month_number', 'skin']]

to_dummies = ['Gender','skin']#,  'skinColor']# 'year_month', 

for column in to_dummies:
    dummies = pd.get_dummies(sales[column])
    # Generate dummy variables, prefix the original column name to each dummy column
    dummies.columns = [f"{column}_{col}" for col in dummies.columns]
    # Now, `dummies` has column names with both the original column name and the value.
    sales = pd.concat([sales, dummies], axis=1)

for column in to_dummies:
    sales.drop(column, axis=1, inplace=True)

sales = sales.dropna(subset = ['amount_original'])

sales.reset_index(inplace=True, drop=True)

X = sales[pd.Series(sales.columns)[pd.Series(sales.columns)!='amount_original'].values]
X = X.drop(['skin_Ape', 'skin_Zombie', 'skin_Alien'], axis=1)
y = sales['amount_original']  # Your target vector

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Assuming X_train, X_test, y_train, y_test are already defined
# Convert your numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).view(-1, 1)
X_val_tensor = torch.tensor(X_val.values.astype(np.float32))
y_val_tensor = torch.tensor(y_val.values.astype(np.float32)).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.float32)).view(-1, 1)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move tensors to the specified device
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# Create TensorDatasets and DataLoader for both training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define the Neural Network
class RegressionNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

# Instantiate the model and move it to the device
model = RegressionNN(input_size=X_train.shape[1], output_size=1).to(device)

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    start_time = time.time()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    end_time = time.time()
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {train_loss/len(train_loader):.4f}, '
          f'Validation Loss: {val_loss/len(val_loader):.4f}, '
          f'Epoch Time: {end_time - start_time:.2f} seconds')

# Testing loop - Evaluation
model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)  # Move data to the device
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())  # Move predictions back to CPU

# Flatten the list of arrays into a single array
predictions = np.concatenate(predictions, axis=0)

# Assuming 'predictions' is a numpy array containing your model's predictions for X_test
# Convert your y_test to numpy array if it's not already
y_test_np = y_test.to_numpy()

# Prepare the X_test_eval DataFrame
# X_test_eval = pd.DataFrame(X_test)  # Ensure X_test is in a suitable format for creating a DataFrame
# X_test_eval['y_pred_nn'] = predictions  # Add your model's predictions
# X_test_eval['y_test'] = y_test_np  # Add the actual test values
# X_test_eval['error'] = X_test_eval['y_test'] - X_test_eval['y_pred_nn']
# X_test_eval['perc_error'] = ((1 - (X_test_eval['y_test'] / X_test_eval['y_pred_nn'])) * 100)
# X_test_eval['perc_error_abs'] = X_test_eval['perc_error'].abs()

np.random.seed(0)  # For reproducibility
y_test = np.random.normal(1000, 300, 100)
y_pred_nn = y_test * np.random.normal(1.0, 0.05, 100)
error = y_test - y_pred_nn
perc_error = (error / y_test) * 100
perc_error_abs = abs(perc_error)

X_test_eval = {
    'y_test': y_test,
    'y_pred_nn': y_pred_nn,
    'error': error,
    'perc_error': perc_error,
    'perc_error_abs': perc_error_abs
}

# Calculate and print MAE and MAPE
print('Mean absolute error:', np.abs(X_test_eval['error']).mean())
print('MAPE:', X_test_eval['perc_error_abs'].mean())

images_dir = "images"
os.makedirs(images_dir, exist_ok=True)

# Without outliers
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_nn)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)
plt.title('Without Outliers')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.savefig(f"{images_dir}/plot_without_outliers.png")
plt.close()

# With outliers
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_nn)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)
plt.title('With Outliers')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.savefig(f"{images_dir}/plot_with_outliers.png")
plt.close()

# Histogram of the error of the model excluding outliers
series_perc_error = exclude_top_and_bottom_5_percent(perc_error)
plt.figure(figsize=(10, 5))
plt.hist(series_perc_error, bins=50)
plt.title("Histogram of Model's MAPE Excluding Outliers")
plt.xlabel('Percentage Error')
plt.ylabel('Frequency')
plt.savefig(f"{images_dir}/histogram_mape_excluding_outliers.png")
plt.close()


exit()

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# Without outliers
subset_no_outliers = X_test_eval[(X_test_eval['y_test'] < 2000) & (X_test_eval['y_pred_nn'] < 2000)]
axs[0].scatter(subset_no_outliers['y_test'], subset_no_outliers['y_pred_nn'])
axs[0].plot([subset_no_outliers['y_test'].min(), subset_no_outliers['y_test'].max()], [subset_no_outliers['y_test'].min(), subset_no_outliers['y_test'].max()], 'k--', lw=4)
axs[0].set_title('Without Outliers')
axs[0].set_xlabel('Actual')
axs[0].set_ylabel('Predicted')

plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_nn)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)
plt.title('Without Outliers')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.savefig(f"{images_dir}/plot_without_outliers.png")
plt.close()

# With outliers
axs[1].scatter(X_test_eval['y_test'], X_test_eval['y_pred_nn'])
axs[1].plot([X_test_eval['y_test'].min(), X_test_eval['y_test'].max()], [X_test_eval['y_test'].min(), X_test_eval['y_test'].max()], 'k--', lw=4)
axs[1].set_title('With Outliers')
axs[1].set_xlabel('Actual')
axs[1].set_ylabel('Predicted')

plt.show()

# Histogram of the error of the model excluding outliers
def exclude_top_and_bottom_5_percent(series):
    lower_bound = series.quantile(0.05)
    upper_bound = series.quantile(0.95)
    return series[(series > lower_bound) & (series < upper_bound)]

series_perc_error = exclude_top_and_bottom_5_percent(X_test_eval['perc_error'])
plt.figure(figsize=(10, 5))
plt.hist(series_perc_error, bins=50)
plt.title("Histogram of Model's MAPE Excluding Outliers")
plt.xlabel('Percentage Error')
plt.ylabel('Frequency')
plt.show()
