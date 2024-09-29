import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

data = pd.read_csv(r'data\train.csv')
data.drop('label', axis=1, inplace=True)

for i in tqdm(range(data.shape[0])):
    raw_data = data.values[i].reshape(28, 28)
    image = Image.fromarray(np.uint8(raw_data))
    image.save(f'./images/image{i}.png')