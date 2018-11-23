from config import build_config
from subprocess import call
from argparse import ArgumentParser
import sys

from mlboardclient.api import client


def main():
    build_config()
    parser = ArgumentParser()
    parser.add_argument('--training_dir')
    parser.add_argument('--research_dir')
    parser.add_argument('--build_id')
    parser.add_argument('--num_steps', default=1000)
    parser.add_argument('--only_train', default='False')
    args, _ = parser.parse_known_args()

    targs = sys.argv[:]
    targs[0] = args.research_dir + '/object_detection/model_main.py'
    targs.insert(0, sys.executable or 'python')
    targs.append("--pipeline_config_path")
    targs.append("faster_rcnn.config")
    targs.append("--model_dir")
    targs.append("%s/%s" % (args.training_dir, args.build_id))

    call(targs)

    client.Client().update_task_info({'train_build_id': args.build_id})


if __name__ == '__main__':
    main()
