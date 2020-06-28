import argparse
import json
import os

import numpy as np
import tensorflow as tf

from npi.task import addition
from npi.model import create_npi_core_train

ENCODERS = {
    'addition': addition.get_encoder(),
}

class EpochLossHistory(tf.keras.callbacks.Callback):
    def on_epoch_begin(self):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))

def process_example(example, max_trace_len, observation_shape, num_arguments, argument_dim, kv_memory_size):
    observation_data = np.zeros((max_trace_len, *observation_shape))
    arguments_data = np.zeros((max_trace_len, num_arguments, argument_dim))
    program_idx_data = np.zeros(max_trace_len)
    stop_data = np.zeros((max_trace_len, 1))
    next_program_idx_data = np.zeros((max_trace_len, kv_memory_size))
    next_arguments_data = np.zeros((max_trace_len, num_arguments, argument_dim))

    observations = example['observations']
    trace = example['trace']
    next_trace = example['trace'][1:] + [(0, (0,) * num_arguments)] # add a dummy program index
    for i, (observation, (program_idx, arguments), (next_program_idx, next_arguments)) in enumerate(zip(observations, trace, next_trace)):
        observation_data[i] = tf.keras.utils.to_categorical(observation, num_classes=observation_shape[-1])
        arguments_data[i] = tf.keras.utils.to_categorical(arguments, num_classes=argument_dim)
        program_idx_data[i] = program_idx
        if i == len(observations) - 1:
            stop_data[i] = 1
        next_program_idx_data[i] = tf.keras.utils.to_categorical(next_program_idx, num_classes=kv_memory_size)
        next_arguments_data[i] = tf.keras.utils.to_categorical(next_arguments, num_classes=argument_dim)

    return {'observation': observation_data, 'program_idx': program_idx_data, 'arguments': arguments_data,
            'stop': stop_data, 'next_program_idx': next_program_idx_data, 'next_arguments': next_arguments_data}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindata', type=str)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint')
    parser.add_argument('--encoder-out', type=str)
    parser.add_argument('--npi-core-out', type=str)

    args = parser.parse_args()

    examples = json.load(open(args.traindata))

    import random
    random.shuffle(examples)

    trace_lens = [len(x['trace']) for x in examples]
    max_trace_len = max(trace_lens)

    # parameters
    num_epochs = 20
    batch_size = 8
    min_samples_per_epoch = 100
    kv_memory_size = 10
    state_dim = 32
    num_arguments = 3
    argument_dim = 10
    mlp_dim = 32
    program_key_dim = 4
    initial_lr = 0.001
    samples_per_decay_period = 10000
    lstm_dim = 256
    num_val = max(int(.1 * len(examples)), 1)
    num_train = len(examples) - num_val
    steps_per_epoch = max(num_train, min_samples_per_epoch)

    # create sub models
    encoder = addition.get_encoder(state_dim=state_dim, num_arguments=num_arguments, argument_dim=argument_dim)
    npi_core = create_npi_core_train(lstm_dim=lstm_dim, program_key_dim=program_key_dim, mlp_dim=mlp_dim, state_dim=state_dim, kv_memory_size=kv_memory_size, argument_dim=argument_dim, num_arguments=num_arguments)
    observation_shape = encoder.get_layer('observation_input_layer').input_shape[2:]
    lstm_dim = npi_core.get_layer('lstm_layer0').output_shape[0][-1]

    # store losses for resampling
    avg_losses = np.ones(num_train)

    all_observation_data = np.zeros((len(examples), max_trace_len, *observation_shape))
    all_arguments_data = np.zeros((len(examples), max_trace_len, num_arguments, argument_dim))
    all_program_idx_data = np.zeros((len(examples), max_trace_len))
    all_lstm_initial_state_data = np.zeros((len(examples), lstm_dim))
    all_stop_data = np.zeros((len(examples), max_trace_len, 1))
    all_next_program_idx_data = np.zeros((len(examples), max_trace_len, kv_memory_size))
    all_next_arguments_data = np.zeros((len(examples), max_trace_len, num_arguments, argument_dim))
    all_sample_weight = np.zeros((len(examples), max_trace_len))
    for i, trace_len in enumerate(trace_lens):
        all_sample_weight[i, :trace_len] = np.ones(trace_len)

    # load data
    for i, example in enumerate(examples):
        processed = process_example(example, max_trace_len, observation_shape, num_arguments, argument_dim, kv_memory_size)
        all_observation_data[i] = processed['observation']
        all_program_idx_data[i] = processed['program_idx']
        all_arguments_data[i] = processed['arguments']
        all_stop_data[i] = processed['stop']
        all_next_program_idx_data[i] = processed['next_program_idx']
        all_next_arguments_data[i] = processed['next_arguments']

    # callbacks
    def lr_schedule_fn(epoch, lr):
        if epoch * samples_per_epoch % samples_per_decay_period == 0:
            # decay
            return lr * .95
        return lr
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_schedule_fn)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.checkpoint_dir, 'model.{epoch:02d}.h5'))
    callbacks = [model_checkpoint]

    # create full training model (encoder + core)
    non_encoder_inputs = [x for x in npi_core.inputs if not x.name.startswith('state_input')]
    outputs = npi_core(encoder.outputs + non_encoder_inputs)
    named_outputs = [tf.keras.layers.Lambda(lambda x: x, name=x.name.split('/')[1])(x) for x in outputs] # remove "model_2/" from name
    model = tf.keras.models.Model(encoder.inputs + non_encoder_inputs, named_outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=initial_lr),
                  loss=['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                  sample_weight_mode='temporal',
                  weighted_metrics=['accuracy'])

    #model = tf.keras.models.load_model('checkpoint/model.20.h5')

    #stop_pred, next_program_idx_pred_oh, next_arguments_pred_oh = model.predict(
    #    [all_observation_data[:num_train], all_arguments_data[:num_train], all_program_idx_data[:num_train], all_lstm_initial_state_data[:num_train]])
    #stop_true, next_program_idx_true_oh, next_arguments_true_oh = [all_stop_data[:num_train], all_next_program_idx_data[:num_train], all_next_arguments_data[:num_train]]

    #next_program_idx_pred = np.argmax(next_program_idx_pred_oh, axis=-1)
    #next_program_idx_true = np.argmax(next_program_idx_true_oh, axis=-1)

    #next_arguments_pred = np.argmax(next_arguments_pred_oh, axis=-1)
    #next_arguments_true = np.argmax(next_arguments_true_oh, axis=-1)

    #for i in range(num_train):
    #    mask = all_sample_weight[i] > 0
    #    if (next_arguments_pred[i][mask] != next_arguments_true[i][mask]).any():
    #        print('mismatch at example', i)
    #        from IPython import embed; embed()

    #return

    model.fit([all_observation_data[:num_train], all_arguments_data[:num_train], all_program_idx_data[:num_train], all_lstm_initial_state_data[:num_train]],
              [all_stop_data[:num_train], all_next_program_idx_data[:num_train], all_next_arguments_data[:num_train]],
              validation_data=([all_observation_data[-num_val:], all_arguments_data[-num_val:], all_program_idx_data[-num_val:], all_lstm_initial_state_data[-num_val:]],
                               [all_stop_data[-num_val:], all_next_program_idx_data[-num_val:], all_next_arguments_data[-num_val:]],
                               [all_sample_weight[-num_val:]] * len(model.outputs)),
              epochs=num_epochs, batch_size=batch_size, callbacks=callbacks, verbose=1, sample_weight=[all_sample_weight[:num_train]] * len(model.outputs),
              steps_per_epoch=None)

    # save model
    encoder.save(args.encoder_out)
    npi_core.save(args.npi_core_out)

    return

    for epoch in range(num_epochs):
        observation_data = np.zeros((samples_per_epoch, max_trace_len, *observation_shape))
        arguments_data = np.zeros((samples_per_epoch, max_trace_len, num_arguments, argument_dim))
        program_idx_data = np.zeros((samples_per_epoch, max_trace_len))
        stop_data = np.zeros((samples_per_epoch, max_trace_len, 1))
        next_program_idx_data = np.zeros((samples_per_epoch, max_trace_len, kv_memory_size))
        next_arguments_data = np.zeros((samples_per_epoch, max_trace_len, num_arguments, argument_dim))
        lstm_initial_state_data = np.zeros((samples_per_epoch, lstm_dim))
        sample_weight = np.zeros((samples_per_epoch, max_trace_len))

        # resample based on weight
        sampled_idxs = [np.random.choice(range(num_train), p=avg_losses / sum(avg_losses)) for _ in range(samples_per_epoch)]
        print('sampled indices/losses', list(zip(sampled_idxs[:10], avg_losses[sampled_idxs[:10]])))

        for i, j in enumerate(sampled_idxs):
            observation_data[i] = all_observation_data[j]
            arguments_data[i] = all_arguments_data[j]
            program_idx_data[i] = all_program_idx_data[j]
            stop_data[i] = all_stop_data[j]
            next_program_idx_data[i] = all_next_program_idx_data[j]
            next_arguments_data[i] = all_next_arguments_data[j]
            sample_weight[i, :trace_lens[j]] = np.ones(trace_lens[j])

        # fit
        model.fit([observation_data, arguments_data, program_idx_data, lstm_initial_state_data], [stop_data, next_program_idx_data, next_arguments_data],
                  validation_data=([all_observation_data[-num_val:], all_arguments_data[-num_val:], all_program_idx_data[-num_val:], all_lstm_initial_state_data[-num_val:]],
                                   [all_stop_data[-num_val:], all_next_program_idx_data[-num_val:], all_next_arguments_data[-num_val:]],
                                   [all_sample_weight[-num_val:]] * len(model.outputs)),
                  epochs=epoch+1, initial_epoch=epoch, batch_size=1, callbacks=callbacks, verbose=1, sample_weight=[sample_weight] * len(model.outputs))

        all_metric_vals = np.zeros((len(examples), 1 + 2 * len(model.outputs)))
        for i, trace_len in enumerate(trace_lens):
            metric_vals = model.evaluate([all_observation_data[np.newaxis, i, :trace_len, :], all_arguments_data[np.newaxis, i, :trace_len, :], all_program_idx_data[np.newaxis, i, :trace_len], all_lstm_initial_state_data[np.newaxis, i]],
                                         [all_stop_data[np.newaxis, i, :trace_len], all_next_program_idx_data[np.newaxis, i, :trace_len, :], all_next_arguments_data[np.newaxis, i, :trace_len, :]],
                                          verbose=0)
            all_metric_vals[i] = metric_vals

        # calculate the true accuracy because the padding skews it
        #print('true train accuracy:', ' '.join('{}: {:.02f}'.format(name, acc) for name, acc in zip(model.output_names, np.mean(all_metric_vals[:num_train], axis=0)[-len(model.outputs):])))
        #print('true val   accuracy:', ' '.join('{}: {:.02f}'.format(name, acc) for name, acc in zip(model.output_names, np.mean(all_metric_vals[-num_val:], axis=0)[-len(model.outputs):])))

        # TODO: reenable after tf.keras callback hooks are deployed to pip
        #loss_history = EpochLossHistory()
        #model.evaluate([observation_data, arguments_data, program_idx_data],
        #               [stop_data, next_program_idx_data, next_arguments_data],
        #               batch_size=1, callbacks=[loss_history])
        #for i in sampled_idxs:
        #    avg_losses[i] = all_metric_vals[i, 0] / trace_lens[i]

    # save model
    encoder.save(args.encoder_out)
    npi_core.save(args.npi_core_out)


if __name__ == '__main__':
    main()