import math
import numpy as np
import pytest
from utils import cartesian_to_sphere, sphere_to_cartesian, adjust_sphere


@pytest.mark.parametrize('test', [
    # azimuth, zenith
    (5.029555, 2.087498),
    (0.417742, 1.549686),
    (1.160466, 2.401942),
    (5.845952, 0.759054)
])
def test_cartesian_to_sphere(test):

    for el in zip(test, adjust_sphere(*cartesian_to_sphere(*sphere_to_cartesian(*test)))):
        assert math.isclose(el[0], el[1], rel_tol=1e-7)
