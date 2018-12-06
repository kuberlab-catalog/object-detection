from config import build_config
from argparse import ArgumentParser
import sys
from mlboardclient.api import client
import tensorflow as tf
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
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    if args.evaluator:
        continuous_eval(estimator, model_dir, eval_input_fns[0],
                        train_steps, 'validation_data')
    else:
        estimator.train(input_fn=train_input_fn, max_steps=train_steps)

def continuous_eval(estimator, model_dir, input_fn, train_steps, name):
    def terminate_eval():
        tf.logging.warning('Eval timeout after 180 seconds of no checkpoints')
        return False

    for ckpt in tf.contrib.training.checkpoints_iterator(
            model_dir, min_interval_secs=180, timeout=None,
            timeout_fn=terminate_eval):

        tf.logging.info('Starting Evaluation.')
        try:
            eval_results = estimator.evaluate(
                input_fn=input_fn, steps=None, checkpoint_path=ckpt, name=name)
            tf.logging.info('Eval results: %s' % eval_results)

            # Terminate eval job when final checkpoint is reached
            current_step = int(os.path.basename(ckpt).split('-')[1])
            if current_step >= train_steps:
                tf.logging.info(
                    'Evaluation finished after training step %d' % current_step)
                break

        except tf.errors.NotFoundError:
            tf.logging.info(
                'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)

if __name__ == '__main__':
    main()