"""Implement NPI core for training and inference using the stateless LSTM that returns sequences and states.
See https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
"""
import tensorflow as tf

def create_npi_core_train(state_dim=32, lstm_dim=256, mlp_dim=32, program_key_dim=4, program_embedding_dim=32, kv_memory_size=10, argument_dim=10, num_arguments=3):
    """Creates a model for training."""

    # step 1: declare layers
    state_input = tf.keras.layers.Input(name='state_input_layer', shape=(None, state_dim))
    program_idx_input = tf.keras.layers.Input(name='program_idx_input_layer', shape=(None,))
    program_embedding_layer = tf.keras.layers.Embedding(name='program_embedding_layer', input_dim=kv_memory_size, output_dim=program_embedding_dim)
    lstm_initial_state_input = tf.keras.layers.Input(name='lstm_inital_state_input_layer', shape=(lstm_dim,))

    # layers to fuse state and program embedding
    concat_layer = tf.keras.layers.Concatenate(name='concat_layer')
    mlp_layer0 = tf.keras.layers.Dense(mlp_dim, name='mlp_layer0', activation='relu')
    mlp_layer1 = tf.keras.layers.Dense(mlp_dim, name='mlp_layer1', activation='relu')

    lstm_layer0 = tf.keras.layers.LSTM(lstm_dim, name='lstm_layer0', return_sequences=True, return_state=True)
    lstm_layer1 = tf.keras.layers.LSTM(lstm_dim, name='lstm_layer1', return_sequences=True, return_state=True)

    stop_layer = tf.keras.layers.Dense(1, name='stop_layer', activation='sigmoid')
    #program_idx_output_layer = tf.keras.layers.Dense(kv_memory_size, name='program_idx_output_layer', activation='softmax')
    program_key_layer = tf.keras.layers.Dense(program_key_dim, name='program_key_layer', activation='relu')
    # rolling my own embedding here
    program_key_embedding_layer = tf.keras.layers.Dense(kv_memory_size, name='program_key_embedding_layer', activation='softmax')
    argument_layers = [tf.keras.layers.Dense(argument_dim, name='argument_layer{}'.format(i), activation='softmax') for i in range(num_arguments)]
    reshape_argument_layers = [tf.keras.layers.Reshape(name='reshape_argument_layer{}'.format(i), target_shape=(-1, 1, argument_dim)) for i in range(num_arguments)]
    arguments_layer = tf.keras.layers.Concatenate(name='arguments_layer', axis=-2)

    # step 2: feed forward
    program_embedding_output = program_embedding_layer(program_idx_input)
    concat_output = concat_layer([state_input, program_embedding_output])
    mlp_layer0_output = mlp_layer0(concat_output)
    mlp_layer1_output = mlp_layer1(mlp_layer0_output)

    lstm_layer0_output, _, _  = lstm_layer0(mlp_layer1_output, initial_state=[lstm_initial_state_input, lstm_initial_state_input])
    lstm_layer1_output, _, _ = lstm_layer1(lstm_layer0_output, initial_state=[lstm_initial_state_input, lstm_initial_state_input])

    stop_layer_output = stop_layer(lstm_layer1_output)
    program_key_layer_output = program_key_layer(lstm_layer1_output)
    program_key_embedding_output = program_key_embedding_layer(program_key_layer_output)
    argument_layer_outputs = [argument_layer(lstm_layer1_output) for argument_layer in argument_layers]
    reshape_argument_layer_outputs = [reshape_argument_layer(argument_layer_output) for reshape_argument_layer, argument_layer_output in zip(reshape_argument_layers, argument_layer_outputs)]
    arguments_layer_output = arguments_layer(reshape_argument_layer_outputs)
    #program_idx_output = program_idx_output_layer(lstm_layer1_output)

    return tf.keras.models.Model(inputs=[state_input, program_idx_input, lstm_initial_state_input],
                                 outputs=[stop_layer_output, program_key_embedding_output, arguments_layer_output])

def create_npi_core_inference(npi_core_train, num_arguments=3):
    """Creates inference model from training model."""
    # step 1: get layers
    state_input_layer = npi_core_train.get_layer('state_input_layer')
    program_idx_input_layer = npi_core_train.get_layer('program_idx_input_layer')
    program_embedding_layer = npi_core_train.get_layer('program_embedding_layer')
    concat_layer = npi_core_train.get_layer('concat_layer')
    mlp_layer0 = npi_core_train.get_layer('mlp_layer0')
    mlp_layer1 = npi_core_train.get_layer('mlp_layer1')

    lstm_layer0 = npi_core_train.get_layer('lstm_layer0')
    lstm_layer1 = npi_core_train.get_layer('lstm_layer1')

    stop_layer = npi_core_train.get_layer('stop_layer')
    program_key_layer = npi_core_train.get_layer('program_key_layer')
    argument_layers = [npi_core_train.get_layer('argument_layer{}'.format(i)) for i in range(num_arguments)]
    reshape_argument_layers = [npi_core_train.get_layer('reshape_argument_layer{}'.format(i)) for i in range(num_arguments)]
    arguments_layer = npi_core_train.get_layer('arguments_layer')

    lstm_dim = lstm_layer0.output_shape[0][-1]

    lstm_layer0_input_h = tf.keras.layers.Input(name='lstm_layer0_input_h', shape=(lstm_dim,))
    lstm_layer0_input_c = tf.keras.layers.Input(name='lstm_layer0_input_c', shape=(lstm_dim,))

    lstm_layer1_input_h = tf.keras.layers.Input(name='lstm_layer1_input_h', shape=(lstm_dim,))
    lstm_layer1_input_c = tf.keras.layers.Input(name='lstm_layer1_input_c', shape=(lstm_dim,))

    #program_idx_output_layer = npi_core_train.get_layer('program_idx_output_layer')

    # step 2: feed forward (again)
    program_embedding_output = program_embedding_layer(program_idx_input_layer.input)
    concat_layer_output = concat_layer([state_input_layer.input, program_embedding_output])
    mlp_layer0_output = mlp_layer0(concat_layer_output)
    mlp_layer1_output = mlp_layer1(mlp_layer0_output)

    lstm_layer0_output, lstm_layer0_h, lstm_layer0_c = lstm_layer0(mlp_layer1_output, initial_state=[lstm_layer0_input_h, lstm_layer0_input_c])
    lstm_layer1_output, lstm_layer1_h, lstm_layer1_c = lstm_layer1(lstm_layer0_output, initial_state=[lstm_layer1_input_h, lstm_layer1_input_c])

    stop_layer_output = stop_layer(lstm_layer1_output)
    program_key_layer_output = program_key_layer(lstm_layer1_output)
    argument_layer_outputs = [argument_layer(lstm_layer1_output) for argument_layer in argument_layers]
    reshape_argument_layer_outputs = [reshape_argument_layer(argument_layer_output) for reshape_argument_layer, argument_layer_output in zip(reshape_argument_layers, argument_layer_outputs)]
    arguments_layer_output = arguments_layer(reshape_argument_layer_outputs)
    #program_idx_output = program_idx_output_layer(lstm_layer1_output)

    return tf.keras.models.Model(inputs=[state_input_layer.input, program_idx_input_layer.input, lstm_layer0_input_h, lstm_layer0_input_c, lstm_layer1_input_h, lstm_layer1_input_c],
                                 outputs=[stop_layer_output, program_key_layer_output, arguments_layer_output] + [lstm_layer0_h, lstm_layer0_c, lstm_layer1_h, lstm_layer1_c])