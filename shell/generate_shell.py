import argparse
import os
import json
from string import Template

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(dir_path + '/parameter.json') as json_file:
    data = json.load(json_file)

with open(dir_path + '/adsbrain_pretrain.template' ) as filein:
    cmd_template = Template( filein.read() )

parameter = data['common']

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp_name', default='', required=True, type=str, help='')
parser.add_argument('--task', default='', required=True, type=str, help='')
parser.add_argument('--criterion', default='', required=True, type=str, help='')
parser.add_argument('--arch', default='', required=True, type=str, help='')
parser.add_argument('--parameter_group', default='', type=str, help='')
parser.add_argument('--restore_file', default='', type=str, help='')
parser.add_argument('--cuda_visible_devices', default='', type=str, help='')
parser.add_argument('--extra_command', default='', type=str, help='')
args = parser.parse_args()


if args.parameter_group:
    for k, v in data[args.parameter_group].items():
        parameter[k] = v

parameter['exp_name']= args.exp_name
parameter['task']= args.task
parameter['criterion']= args.criterion
parameter['arch']= args.arch

if args.cuda_visible_devices:
    parameter['cuda_visible_devices'] = args.cuda_visible_devices

if parameter['cuda_visible_devices']:
    parameter['cuda_visible_devices_command'] = 'CUDA_VISIBLE_DEVICES=' + parameter['cuda_visible_devices'] + ' \\'
else:
    parameter['cuda_visible_devices_command'] = ''

if args.restore_file:
    parameter['restore_file'] = args.restore_file

if parameter['restore_file']:
    parameter['restore_file_command'] = '--restore-file $RESTORE_FILE'
else:
    parameter['restore_file_command'] = ''

parameter['extra_command'] = args.extra_command

result = cmd_template.safe_substitute(parameter)
print(result)
with open(dir_path + '/__adsbrain_pretrain.sh', 'w') as fileout:
    fileout.write(result)
