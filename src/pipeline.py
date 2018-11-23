##
#  Runs train -> eval -> export.
##
import argparse
import logging
import re, sys

from mlboardclient.api import client


SUCCEEDED = 'Succeeded'

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
mlboard = client.Client()
run_tasks = [
    'train',
    'export',
]


def override_task_arguments(task, params):
    for k, v in params.items():
        pattern = re.compile('--{}[ =]([^\s]+|[\'"].*?[\'"])'.format(k))
        resource = task.config['resources'][0]
        task_cmd = resource['command']
        replacement = '--{} {}'.format(k, v)
        if pattern.findall(task_cmd):
            # Replace
            resource['command'] = pattern.sub(
                replacement,
                task_cmd
            )
        else:
            # Add
            val = v
            if isinstance(v, list):
                val = " ".join(v)
            if 'args' in resource:
                resource['args'][k] = val
            else:
                resource['args'] = {k: val}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps')
    parser.add_argument('--model_name')
    parser.add_argument('--model_version')
    parser.add_argument('--resize_min_dimension')
    parser.add_argument('--resize_max_dimension')
    parser.add_argument('--resize_fixed_width')
    parser.add_argument('--resize_fixed_height')
    parser.add_argument('--train_build_id')
    parser.add_argument('--train_checkpoint')
    parser.add_argument('--grid_scales', nargs='*')
    parser.add_argument('--grid_aspect_ratios', nargs='*')
    parser.add_argument('--tf_record_train_path')
    parser.add_argument('--tf_record_test_path')
    parser.add_argument('--label_map_path')
    parser.add_argument('--use_pretrained_checkpoint')
    parser.add_argument('--pretrained_checkpoint_path')
    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    LOG.info(sys.argv)

    override_args = {
        '_common': {
            'grid_scales': args.grid_scales,
            'grid_aspect_ratios': args.grid_aspect_ratios,
            'resize_min_dimension': args.resize_min_dimension,
            'resize_max_dimension': args.resize_max_dimension,
            'resize_fixed_width': args.resize_fixed_width,
            'resize_fixed_height': args.resize_fixed_height,
        },
        'train': {
            'num_steps': args.num_steps,
        },
        'export': {
            'model_name': args.model_name,
            'model_version': args.model_version,
            'train_build_id': '',
            'train_checkpoint': '',
        }
    }

    app = mlboard.apps.get()

    train_build_id = ''
    train_checkpoint = ''

    for task in run_tasks:
        t = app.tasks.get(task)
        if t.name in override_args and override_args[t.name]:
            if 'train_build_id' in override_args[t.name]:
                override_args[t.name]['train_build_id'] = train_build_id
            if 'train_checkpoint' in override_args[t.name]:
                override_args[t.name]['train_checkpoint'] = train_checkpoint
            override_task_arguments(t, override_args[t.name])
        override_task_arguments(t, override_args['_common'])

        LOG.info("Start task %s..." % t.name)

        resource = t.config['resources'][0]
        LOG.info(resource['command'])
        LOG.info(resource['args'])

        started = t.start()

        LOG.info(
            "Run & wait [name=%s, build=%s, status=%s]"
            % (started.name, started.build, started.status)
        )
        completed = started.wait()

        if completed.status != SUCCEEDED:
            LOG.warning(
                "Task %s-%s completed with status %s."
                % (completed.name, completed.build, completed.status)
            )
            LOG.warning(
                'Please take a look at the corresponding task logs'
                ' for more information about failure.'
            )
            LOG.warning("Workflow completed with status ERROR")
            sys.exit(1)

        LOG.info(
            "Task %s-%s completed with status %s."
            % (completed.name, completed.build, completed.status)
        )

        if completed.name == 'train':
            completed.refresh()
            train_build_id = completed.build
            if completed.exec_info and 'train_checkpoint' in completed.exec_info:
                train_checkpoint = completed.exec_info['train_checkpoint']

    LOG.info("Workflow completed with status SUCCESS")


def boolean_string(s):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


if __name__ == '__main__':
    main()