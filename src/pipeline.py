##
#  Runs train -> eval -> export.
##
import argparse
import logging
import re
import sys

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
            if 'args' in resource:
                resource['args'][k] = v
            else:
                resource['args'] = {k: v}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_steps')
    parser.add_argument('--model_name')
    parser.add_argument('--model_version')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    override_args = {
        'train': {
            'num_train_steps': args.num_train_steps,
        },
    }

    app = mlboard.apps.get()

    last_build = ''

    for task in run_tasks:
        t = app.tasks.get(task)
        if t.name in override_args and override_args[t.name]:
            override_task_arguments(t, override_args[t.name])

        r = t.config['resources'][0]
        r['command'] = r['command'].replace('BUILD=1', 'BUILD=%s' % last_build)
        r['command'] = r['command'].replace('CHECKPOINT=1000', 'CHECKPOINT=%s' % args.num_train_steps)
        r['command'] = r['command'].replace('MODEL_NAME=object-detection', 'MODEL_NAME=%s' % args.model_name)
        r['command'] = r['command'].replace('MODEL_VERSION=1.0.0', 'MODEL_VERSION=%s' % args.model_version)

        LOG.info("Start task %s..." % t.name)

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

        last_build = completed.build

    LOG.info("Workflow completed with status SUCCESS")


def boolean_string(s):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


if __name__ == '__main__':
    main()