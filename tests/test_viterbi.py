import numpy as np

from hmm.viterbi import viterbi


def test_three_steps_two_states_by_hand():
    # Log-prob values chosen as exact binary fractions so the hand calculation is exact.
    #
    #   emissions  e0 = [ 0, -1]   e1 = [-2,  0]   e2 = [-5,  0]
    #   transition A (same at every step), A[prev][cur]:  [[ 0, -0.5],
    #                                                      [-3,  0  ]]
    #
    # delta_0 = e0 = [0, -1]
    # t=1: delta_1[0] = -2 + max(0 + 0,    -1 + -3)   = -2 + 0    = -2    back=0
    #      delta_1[1] =  0 + max(0 + -0.5, -1 +  0)   =  0 - 0.5  = -0.5  back=0
    # t=2: delta_2[0] = -5 + max(-2 + 0,  -0.5 + -3)  = -5 - 2    = -7    back=0
    #      delta_2[1] =  0 + max(-2 + -0.5, -0.5 + 0) =  0 - 0.5  = -0.5  back=1
    # best final state = argmax([-7, -0.5]) = 1   (score -0.5)
    # back-track: state2 = 1; back_2[1] = 1 -> state1 = 1; back_1[1] = 0 -> state0 = 0
    # => path [0, 1, 1]. Check by brute force: 0 + -0.5 + 0 + 0 + 0 = -0.5 (the best of 8 paths).
    e = [np.array([0.0, -1.0]), np.array([-2.0, 0.0]), np.array([-5.0, 0.0])]
    A = np.array([[0.0, -0.5], [-3.0, 0.0]])
    assert viterbi(e, lambda t: A) == [0, 1, 1]


def test_matches_brute_force_on_random_problem():
    rng = np.random.default_rng(42)
    e = [rng.normal(size=3) for _ in range(5)]
    A = [None] + [rng.normal(size=(3, 3)) for _ in range(4)]
    best, best_path = -np.inf, None
    for p in np.ndindex(3, 3, 3, 3, 3):
        s = sum(e[t][p[t]] for t in range(5)) + sum(A[t][p[t - 1], p[t]] for t in range(1, 5))
        if s > best:
            best, best_path = s, list(p)
    assert viterbi(e, lambda t: A[t]) == best_path


def test_only_epsilon_transitions_still_returns_full_path():
    eps = np.log(1e-6)
    e = [np.array([-1.0, -3.0, -2.0]), np.array([-4.0, -1.0]), np.array([-2.0, -5.0, -1.0, -9.0])]
    path = viterbi(e, lambda t: np.full((len(e[t - 1]), len(e[t])), eps))
    assert len(path) == 3
    assert path == [0, 1, 2]  # constant transitions: each step just takes its best emission


def test_empty_and_single_step():
    assert viterbi([], lambda t: None) == []
    assert viterbi([np.array([-1.0, -0.5])], lambda t: None) == [1]
