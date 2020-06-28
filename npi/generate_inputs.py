import argparse
import random

from npi.task import addition

def generate_addition_inputs(min_digits, max_digits, examples_per_length):
    assert min_digits >= 1 and min_digits <= max_digits
    for num_digits in range(min_digits, max_digits):
        for _ in range(examples_per_length):
            total = random.randint(1, 10 ** num_digits - 1)
            # randomize number of digits in input0
            max_input0_digits = random.randint(0, addition._num_digits(total))
            input0 = random.randint(0, min(10 ** max_input0_digits, total)) if total else 0
            input1 = total - input0
            assert input0 <= total
            # mix up inputs
            if random.random() > .5:
                input1, input0 = input0, input1
            yield input0, input1

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='task', dest='task')

    addition_parser = subparsers.add_parser('addition')
    addition_parser.add_argument('--min-digits', type=int)
    addition_parser.add_argument('--max-digits', type=int)
    addition_parser.add_argument('--examples-per-length', type=int)

    sorting_parser = subparsers.add_parser('sorting')
    sorting_parser.add_argument('--min-length', type=int)
    sorting_parser.add_argument('--max-length', type=int)

    args = parser.parse_args()

    if args.task == 'addition':
        inputs_list = [x for x in generate_addition_inputs(args.min_digits, args.max_digits, args.examples_per_length)]
    else:
        raise ValueError('invalid task')

    for inputs in inputs_list:
        print(' '.join([str(x) for x in inputs]))

if __name__ == '__main__':
    main()