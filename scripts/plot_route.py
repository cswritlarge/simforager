import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import argparse


matplotlib.rcParams.update({'font.size': 22})


argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--input", required=True, help="Input file for routes")
args = argparser.parse_args()

folder = 'images'
df_list = []
for i in range(64):
    cur_df = pd.read_csv(f'{args.input}/tcell_tracking_{i}.csv').drop_duplicates()
    df_list.append(cur_df)
df = pd.concat(df_list)
df = df.sort_values(by='# time', ascending=True)
ids = df.iloc[:,1].unique()
df_subset = df[df.iloc[:,1] == ids[1]]
# df_subset = df_subset[(df_subset.iloc[:,0] <= 8644)&(df_subset.iloc[:,0] >= 1079)]
print(df_subset.head(20))

st = 0
ed = st+1
for i in range(st, df_subset.shape[0]-1):
    ed = i+1
    if df_subset.iloc[ed-1,0]+1 != df_subset.iloc[ed,0]:
        break
df_subset = df_subset.iloc[st:ed+1, :]

# x = df_subset.iloc[:,2].values
# y = df_subset.iloc[:,3].values

# for i in range(9823, df_subset.shape[0]):
#     print(f'Step {df_subset.iloc[i, 0]}')
#     plt.cla()
#     plt.clf()
#     plt.figure(figsize=(15,15))
#     plt.plot(x[:i+1], y[:i+1], linewidth=1)
#     plt.scatter(x[:i+1], y[:i+1], s=10)
#     plt.scatter(x[i], y[i], s=150, c='black')
#     plt.xlim(np.min(x)-1, np.max(x)+1)
#     plt.ylim(np.min(y)-1, np.max(y)+1)
#     plt.title(f'Step {df_subset.iloc[i, 0]}')
#     plt.savefig(f'{folder}/{df_subset.iloc[i, 0]}.png')
    
    
    
import cv2
import os

folder = 'images'
video_name = 'animation.mp4'
fps = 4

images = [img for img in os.listdir(folder) if img.endswith(".png") or img.endswith(".jpg")]
frame = cv2.imread(os.path.join(folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

for i in range(0, df_subset.shape[0], 50):
    image = f'{folder}/{df_subset.iloc[i, 0]}.png'
    print(f'Load {image}')
    video.write(cv2.imread(image))

video.release()