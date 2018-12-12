from run import export_subprocess
from argparse import ArgumentParser
from config import build_config
from mlboardclient.api import client


def main():

    build_config()

    with open('faster_rcnn.config', 'r') as cf:
        data = cf.read()
        config_html = '<html><head></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">{}</pre></body></html>'.format(
            data)
    client.Client().update_task_info({'#documents.config.html': config_html})

parser = ArgumentParser()
    parser.add_argument('--training_dir')
    parser.add_argument('--research_dir')
    parser.add_argument('--model_name', default="object-detection")
    parser.add_argument('--model_version', default="1.0.0")
    parser.add_argument('--train_build_id')
    parser.add_argument('--train_checkpoint')
    args, _ = parser.parse_known_args()

    export_subprocess(args.research_dir, args.training_dir, args.train_build_id, args.train_checkpoint, args.model_name, args.model_version)


if __name__ == '__main__':
    main()
