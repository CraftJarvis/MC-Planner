import argparse
from pathlib import Path
from tqdm import tqdm
from rich import print
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.json_writer import JsonWriter

'''
split the trajectories into different folders
in_dir: the input directory of json files
out_dir: the output directory of json files
'''
# goal_list = ['log', 'dirt', 'mutton', 'chicken', 'beef', 'string', 'porkchop', 'cobblestone']
def group_by_accomplishments(in_dir, out_dir, extract_feature = False):
    statistic = {}
    reader = JsonReader(in_dir)
    path_dict = {}
    writer_dict = {}
    # for item in goal_list:
    #     path_dict[item] = Path(out_dir, item)
    #     path_dict[item].mkdir(parents=True, exist_ok=True)
    #     print(f"created directory <{path_dict[item]}>. ")
    #     writer_dict[item] = JsonWriter(str(path_dict[item]))
    
    epid = 0
    bar = tqdm(enumerate(reader.read_all_files()))
    for eps, samples in bar:
        for sample in samples.split_by_episode():
            bar.set_description(f'processing samples: {eps}, episode: {epid}')
            epid += 1
            traj_item = None
            complete_accomplishments = []
            for item_list in sample['accomplishments']:
                complete_accomplishments += item_list
                # if len(item_list) > 0 and item_list[0] in goal_list:
                #     traj_item = item_list[0]
                #     statistic[traj_item] = statistic.get(traj_item, 0) + 1
                #     break
            if len(complete_accomplishments) > 0:
                traj_item = complete_accomplishments[-1]
                statistic[traj_item] = statistic.get(traj_item, 0) + 1
            else:
                break
            if traj_item not in writer_dict:
                # lazy create folder
                print(f"found new goal <{traj_item}> creating directory.")
                path_dict[traj_item] = Path(out_dir, traj_item)
                path_dict[traj_item].mkdir(parents=True, exist_ok=True)
                print(f"created directory <{path_dict[traj_item]}>. ")
                writer_dict[traj_item] = JsonWriter(str(path_dict[traj_item]))
                
            if traj_item is not None:
                writer_dict[traj_item].write(sample)
                
    print(f'[√] FINISHED!')
    print("="*80)
    for k, v in statistic.items():
        print(k, v)
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Group the collected data by goals. ")
    parser.add_argument('-i', '--in_dir', default='', type=str)
    parser.add_argument('-o', '--out_dir', default='', type=str)
    args = parser.parse_args()
    print(args)
    # in_dir = '/home/caishaofei/workspace/CODE_BASE/minerl_rllib/demons/two_animals_symbol'
    # out_dir = '/home/caishaofei/workspace/DATA_BASE/two_animals_symbol'
    group_by_accomplishments(args.in_dir, args.out_dir, False)
