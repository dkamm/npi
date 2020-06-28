import tensorflow as tf
import enum
from npi.task import base
from npi.task import util

def _num_digits(x):
    if not x:
        return 1
    num_digits = 0
    while x:
        x //= 10
        num_digits += 1
    return num_digits

class Environment(base.Environment):
    """Environment for addition task."""

    NUM_ROWS = 4
    MAX_COLS = 3000

    class Position(enum.IntEnum):
        INPUT0 = 0
        INPUT1 = 1
        CARRY = 2
        OUTPUT = 3

    POSITIONS = [Position.INPUT0, Position.INPUT1, Position.CARRY, Position.OUTPUT]

    class Action(enum.IntEnum):
        WRITE = 0
        PTR = 1

    ACTIONS = [Action.WRITE, Action.PTR]

    class Direction(enum.IntEnum):
        LEFT = 0
        RIGHT = 1

    DIRECTIONS = [Direction.LEFT, Direction.RIGHT]

    SENTINEL = 10

    def __init__(self, input0, input1):
        self.scratchpad = [[0] * Environment.MAX_COLS for _ in range(Environment.NUM_ROWS)]
        self.positions = [0] * Environment.NUM_ROWS
        answer = input0 + input1
        self.stop_positions = [
            _num_digits(input0),
            _num_digits(input1),
            _num_digits(answer),
            _num_digits(answer),
        ]

        self._load(Environment.Position.INPUT0, input0)
        self._load(Environment.Position.INPUT1, input1)

    def _load(self, row, input):
        if input == 0:
            self.scratchpad[row][0] = 0
            return

        col = 0
        while input:
            self.scratchpad[row][col] = input % 10
            input //= 10
            col += 1

    def observe(self):
        stop_observation = [1 if self.positions[x] >= self.stop_positions[x] else 0 for x in Environment.POSITIONS]
        return [self.scratchpad[x][self.positions[x]] for x in Environment.POSITIONS] + stop_observation

    def step(self, action, arg0, arg1):
        if action == Environment.Action.WRITE:
            if arg0 < 0 or arg0 >= Environment.NUM_ROWS:
                raise ValueError("invalid argument")
            if arg1 < 0 or arg1 > 9:
                raise ValueError("invalid argument")
            self.scratchpad[arg0][self.positions[arg0]] = arg1
        elif action == Environment.Action.PTR:
            if arg0 not in Environment.POSITIONS:
                raise ValueError("invalid argument")
            if arg1 not in Environment.DIRECTIONS:
                raise ValueError("invalid argument")
            if arg1 == Environment.Direction.LEFT:
                self.positions[arg0] += 1
            elif arg1 == Environment.Direction.RIGHT:
                self.positions[arg0] -= 1
            else:
                raise ValueError("invalid argument")
        else:
            raise ValueError("invalid argument")

    def render(self):
        max_cols = max(self.stop_positions[Environment.Position.OUTPUT], 10)
        out = '\n'
        for i in range(Environment.NUM_ROWS):
            position_line = ''
            digit_line = ''
            for j in range(max_cols, -1, -1):
                if j == self.positions[i]:
                    position_line += '^'
                else:
                    position_line += ' '

                digit_line += str(self.scratchpad[i][j])

            out += '|' + digit_line + '|\n'
            out += '|' + position_line + '|\n'
        return out

    def decode_arguments(self, action, arg0, arg1):
        if action == Environment.Action.WRITE:
            arg1str = str(arg1)
        else:
            arg1str = Environment.Direction(arg1).name
        return Environment.Action(action).name, Environment.Position(arg0).name, arg1str

class Subroutine(enum.IntEnum):
    """Addition subroutine enums."""
    ACT = 0
    ADD = 1
    ADD1 = 2
    CARRY = 3
    LSHIFT = 4

def run_reference_program(input0, input1):
    env = Environment(input0, input1)
    observations = []
    traces = []
    _add(env, observations, traces)
    return {'observations': observations, 'trace': traces}

def _add(env, observations, trace):
    observation = env.observe()
    observations.append(observation)
    trace.append((Subroutine.ADD, (0, 0, 0)))
    while env.positions[Environment.Position.OUTPUT] < env.stop_positions[Environment.Position.OUTPUT]:
        _add1(env, observations, trace)
        _lshift(env, observations, trace)

def _add1(env, observations, trace):
    observation = env.observe()
    observations.append(observation)
    trace.append((Subroutine.ADD1, (0, 0, 0)))

    val = 0
    for row in [Environment.Position.INPUT0, Environment.Position.INPUT1, Environment.Position.CARRY]:
        val += env.scratchpad[row][env.positions[row]] % 10

    digit = val % 10

    _act(env, observations, trace, Environment.Action.WRITE, Environment.Position.OUTPUT, digit)

    if val >= 10:
        _carry(env, observations, trace)

def _carry(env, observations, trace):
    observation = env.observe()
    observations.append(observation)
    trace.append((Subroutine.CARRY, (0, 0, 0)))

    _act(env, observations, trace, Environment.Action.PTR, Environment.Position.CARRY, Environment.Direction.LEFT)
    _act(env, observations, trace, Environment.Action.WRITE, Environment.Position.CARRY, 1)
    _act(env, observations, trace, Environment.Action.PTR, Environment.Position.CARRY, Environment.Direction.RIGHT)

def _lshift(env, observations, trace):
    observation = env.observe()
    observations.append(observation)
    trace.append((Subroutine.LSHIFT, (0, 0, 0)))

    for row in Environment.POSITIONS:
        _act(env, observations, trace, Environment.Action.PTR, row, Environment.Direction.LEFT)

def _act(env, observations, trace, action, arg0, arg1):
    observation = env.observe()
    observations.append(observation)
    trace.append((Subroutine.ACT, (action, arg0, arg1)))
    env.step(action, arg0, arg1)

def get_encoder(state_dim=32, num_arguments=3, argument_dim=10):
    observation_input = tf.keras.layers.Input(name='observation_input_layer', shape=(None, 8, 10))
    arguments_input = tf.keras.layers.Input(name='arguments_input_layer', shape=(None, num_arguments, argument_dim))
    reshape_observation_layer = tf.keras.layers.Reshape(name='reshape_observation_layer', target_shape=(-1, 8 * 10))
    reshape_arguments_layer = tf.keras.layers.Reshape(name='reshape_arguments_layer_encoder', target_shape=(-1, num_arguments * argument_dim))
    concat_layer = tf.keras.layers.Concatenate(name='concat_layer')
    hidden_layer0 = tf.keras.layers.Dense(state_dim, name='hidden_layer0', activation='relu')
    hidden_layer1 = tf.keras.layers.Dense(state_dim, name='hidden_layer1', activation='relu')

    reshape_observation_output = reshape_observation_layer(observation_input)
    reshape_arguments_output = reshape_arguments_layer(arguments_input)
    concat_output = concat_layer([reshape_observation_output, reshape_arguments_output])
    hidden_layer0_output = hidden_layer0(concat_output)
    hidden_layer1_output = hidden_layer1(hidden_layer0_output)
    #hidden_output = hidden_layer(reshape_observation_output)

    return tf.keras.models.Model([observation_input, arguments_input], hidden_layer1_output)
    #return tf.keras.models.Model([observation_input, arguments_input], hidden_output)
