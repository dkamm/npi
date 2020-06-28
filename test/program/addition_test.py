import numpy as np

from npi.environment.addition import AdditionEnvironment
from npi.program.addition import AdditionProgram

def test_basic():
    env = AdditionEnvironment(96, 125)
    prog = AdditionProgram()

    result = prog.run(env)

    assert len(result['observations']) == 30
    assert len(result['observations']) == len(result['trace'])

    np.testing.assert_array_equal([3,3,3,3], env.positions)

    exp = np.array([
        [6,9,0],
        [5,2,1],
        [0,1,1],
        [1,2,2],
    ])
    actual = np.array(env.scratchpad)
    np.testing.assert_array_equal(exp, actual[:,:3])
    np.testing.assert_array_equal(0, actual[:, 3:])