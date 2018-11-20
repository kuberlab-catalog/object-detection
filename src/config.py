import argparse, jinja2, sys


def main():
    build_config()


def build_config():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--num_steps', default=1000)
    parser.add_argument('--resize_min_dimension', type=positive_int, default=600)
    parser.add_argument('--resize_max_dimension', type=positive_int, default=1024)
    parser.add_argument('--resize_fixed_width', type=positive_int, default=0)
    parser.add_argument('--resize_fixed_height', type=positive_int, default=0)
    parser.add_argument('--grid_scales', nargs='+', default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument('--grid_aspect_ratios', nargs='+', default=[0.5, 1.0, 2.0])
    args, _ = parser.parse_known_args()

    targs = {
        'data_dir': args.data_dir,
        'num_steps': args.num_steps,
        'resize_min_dimension': args.resize_min_dimension,
        'resize_max_dimension': args.resize_max_dimension,
        'resize_fixed_width': args.resize_fixed_width,
        'resize_fixed_height': args.resize_fixed_height,
        'grid_scales': args.grid_scales,
        'grid_aspect_ratios': args.grid_aspect_ratios,
    }

    if targs['resize_min_dimension'] == 0 or targs['resize_min_dimension'] == 0:
        targs['resize_min_dimension'] = 0
        targs['resize_max_dimension'] = 0
    if targs['resize_fixed_width'] == 0 or targs['resize_fixed_height'] == 0:
        targs['resize_fixed_width'] = 0
        targs['resize_fixed_height'] = 0

    if targs['resize_min_dimension'] == 0 and targs['resize_fixed_width'] == 0:
        raise Exception('it should be set positive resize_min_dimension with resize_max_dimension or resize_fixed_width with resize_fixed_height')
    if targs['num_steps'] == 0:
        raise Exception('num_steps must be positive')
    if targs['data_dir'] == '':
        raise Exception('data_dir is not set')

    cfg_file = "faster_rcnn_resnet101_pets.config"
    t = open(cfg_file, "r")
    template = jinja2.Template(t.read())
    t.close()
    tw = open(cfg_file, "w+")
    cfg = str(template.render(args=targs))
    tw.write(cfg)
    tw.close()


def positive_int(val):
    ival = 0
    try:
        ival = int(val)
    except:
        pass
    finally:
        return ival if ival > 0 else 0


if __name__ == '__main__':
    main()
