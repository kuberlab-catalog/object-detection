from config import build_config
from argparse import ArgumentParser
import sys
from mlboardclient.api import client
import tensorflow as tf
def main():
    targs = build_config()
    parser = ArgumentParser()
    parser.add_argument('--training_dir')
    parser.add_argument('--research_dir')
    parser.add_argument('--build_id')
    parser.add_argument('--only_train', default='False')
    parser.add_argument('--model_name')
    parser.add_argument('--model_version')
    args, _ = parser.parse_known_args()

    with open('faster_rcnn.config','r') as cf:
        data = cf.read()
        config_html = '<html><head></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">{}</pre></body></html>'.format(data)

    client.Client().update_task_info({'#documents.config.html':config_html})


    sys.path.append(args.research_dir)
    from object_detection import model_lib
    from object_detection import model_hparams
    num_steps = targs['num_steps']
    model_dir = '{}/{}'.format(args.training_dir, args.build_id)
    config = tf.estimator.RunConfig(model_dir=model_dir)
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(None),
        pipeline_config_path='faster_rcnn.config',
        train_steps=num_steps,
        sample_1_of_n_eval_examples=1,
        sample_1_of_n_eval_on_train_examples=(5))
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    train_steps = train_and_eval_dict['train_steps']
    estimator.train(input_fn=train_input_fn, max_steps=train_steps)

if __name__ == '__main__':
    main()