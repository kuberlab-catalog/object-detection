from config import build_config
from subprocess import check_output
from argparse import ArgumentParser
import sys

from mlboardclient.api import client


def main():
    build_config()
    parser = ArgumentParser()
    parser.add_argument('--training_dir')
    parser.add_argument('--research_dir')
    parser.add_argument('--build_id')
    parser.add_argument('--num_steps')
    args, _ = parser.parse_known_args()

    targs = sys.argv[:]
    targs[0] = args.research_dir + '/object_detection/model_main.py'
    targs.insert(0, sys.executable or 'python')
    targs.append("--pipeline_config_path")
    targs.append("faster_rcnn_resnet101_pets.config")
    targs.append("--model_dir")
    targs.append("%s/%s" % (args.training_dir, args.build_id))

    print(check_output(targs))

    client.Client().update_task_info({
        'train_build_id': args.build_id,
        'train_checkpoint': args.num_steps,
    })


if __name__ == '__main__':
    main()
