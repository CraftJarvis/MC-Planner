import argparse
from pathlib import Path
from tqdm import tqdm
from rich import print
from ray.rllib.offline.json_reader import JsonReader
import multiprocessing as mp
import os
import lmdb
import uuid
import pickle
import json
import random


class ReaderWorker(mp.Process):
    def __init__(self, pipe):
        super().__init__()

        self.pipe = pipe

    def run(self):

        while True:

            command, args = self._recv_message()

            if command == "new_json":
                json_file_name = args

                self._send_message("json_received")

                reader = JsonReader(json_file_name)
                for traj in reader.read_all_files():
                    self._send_message("send_traj", traj)

                    command, _ = self._recv_message()
                    assert command == "received_traj"

                self._send_message("file_completed")

            elif command == "kill":
                return

    def _send_message(self, command, args = None):
        self.pipe.send((command, args))

    def _recv_message(self):
        self.pipe.poll(None) # wait until new message is received
        command, args = self.pipe.recv()

        return command, args


def main():
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('-i', '--in-dir', default = '/home/anji/Documents/projects/minerl/vpt/forest_many_tasks', type = str)
    parser.add_argument('-o', '--out-dir', default = './recordings', type = str)
    parser.add_argument('-n', '--num-workers', default = 8, type = int)
    parser.add_argument('--lmdb-chunk-size', default = 8, type = int)
    pargs = parser.parse_args()

    lmdb_trajs_path = os.path.join(pargs.out_dir, "trajs")
    lmdb_indices_path = os.path.join(pargs.out_dir, "indices")
    if not os.path.exists(pargs.out_dir):
        os.mkdir(pargs.out_dir)
    if not os.path.exists(lmdb_trajs_path):
        os.mkdir(lmdb_trajs_path)
    if not os.path.exists(lmdb_indices_path):
        os.mkdir(lmdb_indices_path)

    trajs_lmdb_env = lmdb.open(lmdb_trajs_path, map_size = 1099511627776)
    indices_lmdb_env = lmdb.open(lmdb_indices_path, map_size = 1073741824)

    cnt = 0
    
    json_files = []
    goal_folders = os.listdir(pargs.in_dir)
    for goal_folder in goal_folders:
        goal_path = os.path.join(pargs.in_dir, goal_folder)
        if not os.path.isdir(goal_path):
            continue
            
        files = os.listdir(goal_path)
        for file in files:
            json_files.append(os.path.join(goal_path, file))
    random.shuffle(json_files)

    workers = []
    pipes = []
    for _ in range(pargs.num_workers):
        parent_pipe, child_pipe = mp.Pipe()
        pipes.append(parent_pipe)

        worker = ReaderWorker(child_pipe)
        workers.append(worker)

    for worker in workers:
        worker.start()

    def _send_message(worker_id, command, args = None):
        pipes[worker_id].send((command, args))

    def _recv_message_nonblocking(worker_id):
        if not pipes[worker_id].poll():
            return None, None
        command, args = pipes[worker_id].recv()
        return command, args

    def _recv_message(worker_id):
        pipes[worker_id].poll(None) # wait until new message is received
        command, args = pipes[worker_id].recv()

        return command, args

    worker_idle = [False for _ in range(pargs.num_workers)]
    for worker_id in range(pargs.num_workers):
        file = json_files.pop()
        _send_message(worker_id, "new_json", file)

    while (not all(worker_idle)) or len(json_files) > 0:
        if any(worker_idle) and len(json_files) > 0:
            worker_id = worker_idle.index(True)
            file = json_files.pop()
            _send_message(worker_id, "new_json", file)
            command, _ = _recv_message(worker_id)
            assert command == "json_received"

            worker_idle[worker_id] = False

        for worker_id in range(pargs.num_workers):
            command, args = _recv_message_nonblocking(worker_id)

            if command == "send_traj":
                _send_message(worker_id, "received_traj")
                sample_batch = args

                traj = []
                for step_id in range(len(sample_batch)):
                    traj.append((
                        sample_batch["obs"][step_id], 
                        sample_batch["action"][step_id], 
                        sample_batch["reward"][step_id], 
                        sample_batch["done"][step_id],
                        sample_batch["info"][step_id]
                    ))
                # append a fake last state
                traj.append((sample_batch["obs"][-1], None, None, None, None))

                # get reward and accomplishments
                cum_reward = sum([item[2] for item in traj[:-1]])
                accomplishments = set()
                for item in traj[:-1]:
                    for accomplishment in item[4]['accomplishments']:
                        accomplishments.add(accomplishment)
                accomplishments = list(accomplishments)
                assert cum_reward > 0

                traj_name = str(uuid.uuid1())
                chunks = []
                traj_len = len(traj)
                for chunk_start in range(0, traj_len, pargs.lmdb_chunk_size):
                    chunk_end = min(chunk_start + pargs.lmdb_chunk_size, traj_len)

                    serialized_chunk = pickle.dumps(traj[chunk_start:chunk_end])
                    chunks.append(serialized_chunk)

                # Write indices
                txn = indices_lmdb_env.begin(write = True)
                traj_info = {"num_chunks": len(chunks), "accomplishments": accomplishments, "horizon": traj_len}
                serialized_traj_info = json.dumps(traj_info, indent=2).encode()
                txn.put(traj_name.encode(), serialized_traj_info)
                txn.commit()

                # Write chunks
                txn = trajs_lmdb_env.begin(write = True)
                for i, chunk in enumerate(chunks):
                    key = traj_name + "_" + str(i)
                    txn.put(key.encode(), chunk)
                txn.commit()
                cnt += 1
                print("Dumped 1 trajectory of length {} - {} - total: {}".format(traj_len, accomplishments, cnt))

            elif command == "file_completed":
                worker_idle[worker_id] = True

                if len(json_files) == 0:
                    _send_message(worker_id, "kill")


if __name__ == '__main__':
    mp.set_start_method("forkserver")
    main()
