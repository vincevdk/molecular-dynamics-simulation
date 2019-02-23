import numpy as np
from simulation import calculate_minimal_distance_and_direction, calculate_force
import pytest

def test_2_particles_minimal_distance_and_direction():
    particles = 2 
    pos_at_t = np.array([[1,1,1],[9,9,9]])
    min_dis, min_dir = calculate_minimal_distance_and_direction(particles, pos_at_t)
    
    assert np.all(min_dir[0] == np.array([[0,0,0],[-2,-2,-2]]))
    assert np.all(min_dir[1] == np.array([[2,2,2],[0,0,0]]))
    assert min_dis[0,0,0] == pytest.approx(0, 0.1)
    assert min_dis[0,1,0] == pytest.approx(3.4, 0.1)

def test_multiple_particles_minimal_distance_and_direction():
    particles = 4
    pos_at_t = np.array([[1,1,1],[9,9,9],[2,2,2],[4,4,4]])
    min_dis, min_dir = calculate_minimal_distance_and_direction(particles, 
                                                                pos_at_t)
    assert np.all(min_dir[0,0,0] == np.array([0,0,0]))
    assert np.all(min_dir[1,0] == np.array([2,2,2]))
    assert min_dis[0,0,0] == pytest.approx(0, 0.1)
    assert min_dis[0,1,0] == pytest.approx(3.4, 0.1)

def test_calcualte_force():
    particles = 4
    pos_at_t = np.array([[1,1,1],[9,9,9],[2,2,2],[4,4,4]])
    min_dis, min_dir = calculate_minimal_distance_and_direction(particles,
                                                                pos_at_t)

    F = calculate_force(min_dir,min_dis)
    assert np.shape(F) == (4,4,3)
