from config import build_config
from subprocess import call
from argparse import ArgumentParser
import sys, os

from mlboardclient.api import client


def main():
    build_config()
    parser = ArgumentParser()
    parser.add_argument('--research_dir')
    parser.add_argument('--training_dir')
    parser.add_argument('--model_name', default="object-detection")
    parser.add_argument('--model_version', default="1.0.0")
    parser.add_argument('--train_build_id')
    parser.add_argument('--train_checkpoint')
    args, _ = parser.parse_known_args()

    targs = sys.argv[:]

    targs[0] = args.research_dir + '/object_detection/export_inference_graph.py'
    targs.insert(0, sys.executable or 'python')

    targs.append("--pipeline_config_path")
    targs.append("faster_rcnn.config")

    targs.append("--trained_checkpoint_prefix")
    targs.append("%s/%s/model.ckpt-%s" % (args.training_dir, args.train_build_id, args.train_checkpoint))

    targs.append("--output_directory")
    targs.append("%s/model/%s" % (args.training_dir, args.train_build_id))

    targs.append("--input_type")
    targs.append("encoded_image_string_tensor")

    print("Execute: ", targs)
    call(targs)

    m = client.Client()
    m.model_upload(
        args.model_name,
        args.model_version,
        '%s/model/%s/saved_model' % (args.training_dir, args.train_build_id),
    )
    m.update_task_info({
        'model': '#/%s/catalog/mlmodel/%s/versions/%s' % (
            os.environ['WORKSPACE_NAME'],
            args.model_name,
            args.model_version,
        ),
    })


if __name__ == '__main__':
    main()
