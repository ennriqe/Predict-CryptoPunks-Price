from datasets import load_dataset
from PIL import Image
import io
import numpy as np
from copy import deepcopy
from utils.utils import extract_id, get_most_common_color
import pandas as pd

if __name__ == "__main__":
    x = set()
    dataset = load_dataset("huggingnft/cryptopunks")
    df = pd.DataFrame()
    for i, image_data in enumerate(dataset['train']):
        if i%100 == 0:
            print(i)
        image, id, url = image_data['image'], image_data['id'], image_data['image_original_url']
        cpunks_id = extract_id(url)
        skin_color = get_most_common_color(image)
        df = df.append({'id': cpunks_id, 'skin_color': skin_color, 'url': url}, ignore_index=True)
    print(df.head(5))
    df.to_csv("cryptopunks_skin.csv", index=False)
