import argparse
from pathlib import Path
from tqdm import tqdm
from rich import print
from ray.rllib.offline.json_reader import JsonReader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute statistics of collected data. ")
    parser.add_argument('-i', '--in_dir', default='', type=str)
    args = parser.parse_args()
    print(args)
    root_path = Path(args.in_dir).glob('*')
    statistic = {}
    for sub_path in root_path:
        name = sub_path.name
        reader = JsonReader(str(sub_path))
        cnt = 0 
        for batch in reader.read_all_files():
            cnt += 1
        print(f"{name}: {cnt}")
        statistic[name] = cnt
    print(statistic)
    print(f"total: {sum(list(statistic.values()))}")
