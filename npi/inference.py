import argparse
import json

import numpy as np
import tensorflow as tf

from npi.task import addition
from npi.model import create_npi_core_inference

def run(env, encoder, npi_core, program_key_embedding, initial_subroutine, act_subroutine, max_steps=100):
    non_encoder_inputs = [x for x in npi_core.inputs if not x.name.startswith('state_input')]
    outputs = npi_core(encoder.outputs + non_encoder_inputs)
    model = tf.keras.models.Model(encoder.inputs + non_encoder_inputs, outputs)

    observation_dim = encoder.get_layer('observation_input_layer').input_shape[-1]
    num_arguments, argument_dim = encoder.get_layer('arguments_input_layer').input_shape[-2:]
    print(num_arguments, argument_dim)
    lstm_layer0 = npi_core.get_layer('lstm_layer0')
    lstm_layer1 = npi_core.get_layer('lstm_layer1')

    observations = []
    trace = []

    # initialize inputs
    observation = env.observe()
    arguments = (0,) * num_arguments
    program_idx = initial_subroutine
    lstm_layer0_h = np.zeros((1, lstm_layer0.units))
    lstm_layer0_c = np.zeros((1, lstm_layer0.units))
    lstm_layer1_h = np.zeros((1, lstm_layer1.units))
    lstm_layer1_c = np.zeros((1, lstm_layer1.units))

    stop = 0
    steps = 0

    while steps < max_steps:
        steps += 1
        observations.append(observation)
        print(type(program_idx), type(arguments[0]))
        trace.append((program_idx, arguments))

        # feed forward
        stop_layer_output, program_key_output, arguments_output, lstm_layer0_h, lstm_layer0_c, lstm_layer1_h, lstm_layer1_c = model.predict(
            [tf.keras.utils.to_categorical(observation, observation_dim)[np.newaxis, np.newaxis, :],
             tf.keras.utils.to_categorical(arguments, argument_dim)[np.newaxis, np.newaxis, :],
             np.array([program_idx])[np.newaxis, :],
             lstm_layer0_h,
             lstm_layer0_c,
             lstm_layer1_h,
             lstm_layer1_c])

        print('in ', program_idx, arguments, observation)

        # extract fields
        stop = stop_layer_output[0][0]
        arguments = tuple(map(int, np.argmax(arguments_output[0][0], axis=-1)))
        program_key = program_key_output[0][0]
        program_idx = int(np.argmax(np.dot(program_key, program_key_embedding)))
        observation = env.observe() # observe before action is applied to use for inference

        if stop > .5:
            break

        if program_idx == act_subroutine:
            env.step(*list(arguments))

    if steps > max_steps:
        print('max steps hit')
    return {'observations': observations, 'trace': trace}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder-path', type=str)
    parser.add_argument('--npi-core-path', type=str)
    parser.add_argument('--outpath', type=str)
    subparsers = parser.add_subparsers(title='task', dest='task')
    addition_parser = subparsers.add_parser('addition')
    addition_parser.add_argument('--input0', type=int)
    addition_parser.add_argument('--input1', type=int)

    args = parser.parse_args()

    if args.task == 'addition':
        env = addition.Environment(args.input0, args.input1)
        initial_subroutine = addition.Subroutine.ADD
        act_subroutine = addition.Subroutine.ACT
    else:
        raise ValueError('invalid task')

    encoder = tf.keras.models.load_model(args.encoder_path)
    npi_core_train = tf.keras.models.load_model(args.npi_core_path)
    program_key_embedding = npi_core_train.get_layer('program_key_embedding_layer').get_weights()[0]
    npi_core_inference = create_npi_core_inference(npi_core_train)

    example = run(env, encoder, npi_core_inference, program_key_embedding, initial_subroutine, act_subroutine)
    print(env.render())
    with open(args.outpath, 'w') as fh:
        json.dump(example, fh)

if __name__ == '__main__':
    main()
