import argparse
import json
import sys

from npi.task import addition

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin, help='read space separated inputs, one per line')
    parser.add_argument('--task', choices=['addition'])
    args = parser.parse_args()

    inputs_list = [tuple(int(x) for x in l.split(' ')) for l in args.inputfile]

    if args.task == 'addition':
        module = addition
    else:
        raise ValueError('invalid task')

    examples = []
    for inputs in inputs_list:
        example = module.run_reference_program(*inputs)
        example['inputs'] = inputs
        example['task'] = args.task
        examples.append(example)

    print(json.dumps(examples, indent=2))

if __name__ == '__main__':
    main()
