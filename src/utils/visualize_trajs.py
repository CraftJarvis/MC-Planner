import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from rich import print
from ray.rllib.offline.json_reader import JsonReader

def draw_eps(reader, goal, e):
    sample = reader.next()
    video_path = f"./log/{goal}_{e}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # print(sample['obs'][0]['rgb'].shape)
    out = cv2.VideoWriter(video_path, fourcc, 40.0, (160, 120))
    for i in tqdm(range(sample['obs'].shape[0]), leave=False):
        out.write(cv2.cvtColor(sample['obs'][i]['rgb'], cv2.COLOR_RGB2BGR))
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the collected grouped trajectories. ")
    parser.add_argument('-i', '--in_dir', default='', type=str)
    parser.add_argument('-g', '--goal', default='log', type=str)
    parser.add_argument('-n', '--num', default=50, type=int)
    args = parser.parse_args()
    print(args)
    
    reader = JsonReader(str(Path(args.in_dir,args.goal)))
    for e in range(args.num):
        draw_eps(reader, args.goal, e)
