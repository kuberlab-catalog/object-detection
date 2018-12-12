from config import build_config, str_bool
from argparse import ArgumentParser
import sys, os, numbers
from mlboardclient.api import client
import tensorflow as tf
from object_detection import model_lib, model_hparams, exporter
from subprocess import call

from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

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
    parser.add_argument('--export', type=str_bool, help='Export model')
    parser.add_argument('--model_name')
    parser.add_argument('--model_version')
    args, _ = parser.parse_known_args()

    with open('faster_rcnn.config', 'r') as cf:
        data = cf.read()
        config_html = '<html><head></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">{}</pre></body></html>'.format(
            data)

    client.Client().update_task_info({'#documents.config.html': config_html})

    sys.path.append(args.research_dir)
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
        tf.logging.info('Starting Evaluation.')
        model_name = None
        model_version = None
        if args.export:
            model_name = args.model_name
            model_version = args.model_version
        continuous_eval(estimator, model_dir, eval_input_fns[0], 'validation_data', args, model_name, model_version)
    elif os.environ.get("TF_CONFIG", '') != '':
        tf.logging.info('Starting Distributed.')
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
        tf.logging.info('Starting Training.')
        estimator.train(input_fn=train_input_fn, max_steps=train_steps)


def continuous_eval(estimator, model_dir, input_fn, name, args, model_name=None, model_version=None):
    def terminate_eval():
        tf.logging.warning('Eval timeout after 180 seconds of no checkpoints')
        return False

    loss = None
    for ckpt in tf.contrib.training.checkpoints_iterator(
            model_dir, min_interval_secs=180, timeout=None,
            timeout_fn=terminate_eval):

        tf.logging.info('Starting Evaluation.')
        try:
            eval_results = estimator.evaluate(
                input_fn=input_fn, steps=None, checkpoint_path=ckpt, name=name)
            res = {}
            for k, v in eval_results.items():
                if isinstance(v, numbers.Number):
                    res[k] = v
                if k == 'Loss/total_loss':
                    tf.logging.info('!!!!! Previous loss: {}, current: {}'.format(loss, v))
                    if (loss is None) or loss > v:
                        if model_name is not None and model_version is not None:
                            current_step = int(os.path.basename(ckpt).split('-')[1])
                            # export(args.training_dir, args.build_id, current_step, model_name, model_version)
                            # try to not export on 0 checkpoint
                            # if loss is not None:
                            tf.logging.info('!!!!! Start exporting, step: {}'.format(current_step))
                            export_subprocess(
                                args.research_dir, args.training_dir, args.build_id,
                                current_step, model_name, model_version,
                            )
                            # else:
                            #     tf.logging.info('!!!!! Skip exporting, step: {}'.format(current_step))
                            loss = v
                            tf.logging.info('!!!!! New loss value: {}'.format(loss))
                        else:
                            tf.logging.info('!!!!! Skipping model export')
            tf.logging.info('Eval results: {}'.format(res))

            # Terminate eval job when final checkpoint is reached
            # current_step = int(os.path.basename(ckpt).split('-')[1])

        except tf.errors.NotFoundError:
            tf.logging.info(
                'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)


def export(training_dir, train_build_id, train_checkpoint, model_name, model_version):

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile('faster_rcnn.config', 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    tf.logging.info('!!!!! Continue exporting, train_checkpoint: {}'.format(train_checkpoint))
    res = exporter.export_inference_graph(
        'encoded_image_string_tensor', pipeline_config,
        '{}/{}/model.ckpt-{}'.format(training_dir, train_build_id, train_checkpoint),
        '{}/model/{}'.format(training_dir, train_build_id),
        # write_inference_graph=FLAGS.write_inference_graph,
    )

    tf.logging.info('!!!!! Export result: {}'.format(res))

    after_export(training_dir, train_build_id, model_name, model_version)


def export_subprocess(research_dir, training_dir, train_build_id, train_checkpoint, model_name, model_version):

    targs = sys.argv[:]

    targs[0] = research_dir + '/object_detection/export_inference_graph.py'
    targs.insert(0, sys.executable or 'python')

    targs.append("--pipeline_config_path")
    targs.append("faster_rcnn.config")

    targs.append("--trained_checkpoint_prefix")
    targs.append("%s/%s/model.ckpt-%s" % (training_dir, train_build_id, train_checkpoint))

    targs.append("--output_directory")
    targs.append("%s/model/%s" % (training_dir, train_build_id))

    targs.append("--input_type")
    targs.append("encoded_image_string_tensor")

    tf.logging.info('!!!!! Export subprocess start: {}'.format(targs))

    res = call(targs)

    tf.logging.info('!!!!! Export subprocess result: {}'.format(res))


    # args = [
    #     sys.executable or 'python',
    #     research_dir + '/object_detection/export_inference_graph.py',
    #
    #     '--input_type',
    #     'encoded_image_string_tensor',
    #
    #     '--pipeline_config_path',
    #     'faster_rcnn.config',
    #
    #     '--trained_checkpoint_prefix',
    #     '%s/%s/model.ckpt-%s' % (training_dir, train_build_id, train_checkpoint),
    #
    #     '--output_directory',
    #     '%s/model/%s' % (training_dir, train_build_id),
    # ]
    # tf.logging.info('!!!!! Export subprocess start: {}'.format(args))
    #
    # res = call(args)
    #
    # tf.logging.info('!!!!! Export subprocess result: {}'.format(res))

    after_export(training_dir, train_build_id, model_name, model_version)


def after_export(training_dir, train_build_id, model_name, model_version):

    m = client.Client()
    m.model_upload(
        model_name,
        model_version,
        '{}/model/{}/saved_model'.format(training_dir, train_build_id),
    )
    m.update_task_info({
        'model': '#/%s/catalog/mlmodel/%s/versions/%s' % (
            os.environ['WORKSPACE_NAME'],
            model_name,
            model_version,
        ),
    })



if __name__ == '__main__':
    main()
