import numpy as np
from simulation import calculate_minimal_distance_and_direction
import pytest

def test_minimal_distance_and_direction():
    particles = 2 
    pos_at_t = np.array([[1,1,1],[9,9,9]])
    min_dis, min_dir = calculate_minimal_distance_and_direction(particles, pos_at_t)
    print(min_dir[0])
    assert np.all(min_dir[0] == np.array([[0,0,0],[-2,-2,-2]]))
    assert np.all(min_dir[1] == np.array([[2,2,2],[0,0,0]]))
    assert min_dis[0] == pytest.approx(3.4, 0.1)


