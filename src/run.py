from config import build_config
from argparse import ArgumentParser
import sys
from mlboardclient.api import client
import tensorflow as tf
import os
import numbers


def main():
    targs = build_config()
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(worker=False)
    group.set_defaults(evaluator=False)
    group.add_argument('--worker', dest='worker', action='store_true',
                       help='Training')
    group.add_argument('--evaluator', dest='evaluator', action='store_true',
                       help='Continuously evaluate model')
    parser.add_argument('--training_dir')
    parser.add_argument('--research_dir')
    parser.add_argument('--build_id')
    parser.add_argument('--only_train', default='False')
    parser.add_argument('--model_name')
    parser.add_argument('--model_version')
    parser.add_argument('--export', action='store_true', help='Export model')
    args, _ = parser.parse_known_args()

    with open('faster_rcnn.config', 'r') as cf:
        data = cf.read()
        config_html = '<html><head></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">{}</pre></body></html>'.format(
            data)

    client.Client().update_task_info({'#documents.config.html': config_html})

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
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    if args.evaluator:
        model_name = None
        model_version = None
        if args.export:
            model_name = args.model_name
            model_version = args.model_version
        continuous_eval(estimator, model_dir, eval_input_fns[0], 'validation_data', model_name, model_version)
    elif os.environ.get("TF_CONFIG", '') != '':
        eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
        predict_input_fn = train_and_eval_dict['predict_input_fn']
        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
    else:
        estimator.train(input_fn=train_input_fn, max_steps=train_steps)


def continuous_eval(estimator, model_dir, input_fn, name, model_name=None, model_version=None):
    def terminate_eval():
        tf.logging.warning('Eval timeout after 180 seconds of no checkpoints')
        return False

    for ckpt in tf.contrib.training.checkpoints_iterator(
            model_dir, min_interval_secs=180, timeout=None,
            timeout_fn=terminate_eval):

        tf.logging.info('Starting Evaluation.')
        loss = None
        try:
            eval_results = estimator.evaluate(
                input_fn=input_fn, steps=None, checkpoint_path=ckpt, name=name)
            ##names = ['DetectionBoxes_Precision/mAP','DetectionBoxes_Precision/mAP (large)','']
            res = {}
            for k, v in eval_results.items():
                if isinstance(v, numbers.Number):
                    res[k] = v
                if k == 'Loss/total_loss':
                    tf.logging.info('Previous loss: {}, current: {}'.format(loss, v))
                    if loss is None or loss < v:
                        if model_name is not None and model_version is not None:
                            tf.logging.info('Starting export to model {}:{}'.format(model_name, model_version))
                        else:
                            tf.logging.info('Skipping model export')
            tf.logging.info('Eval results: {}'.format(res))

            # Terminate eval job when final checkpoint is reached
            current_step = int(os.path.basename(ckpt).split('-')[1])

        except tf.errors.NotFoundError:
            tf.logging.info(
                'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)


if __name__ == '__main__':
    main()
