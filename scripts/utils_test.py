import math
import numpy as np
import pytest #type: ignore
from scripts.utils import cartesian_to_sphere, sphere_to_cartesian, adjust_sphere, convert_bytes_to_gmbkb


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



def test_convert_bytes_to_gmbkb():
    assert convert_bytes_to_gmbkb(1024) == "1.00 KB"
    assert convert_bytes_to_gmbkb(1048576) == "1.00 MB"
    assert convert_bytes_to_gmbkb(1073741824) == "1.00 GB"
    assert convert_bytes_to_gmbkb(1099511627776) == "1.00 TB"
    assert convert_bytes_to_gmbkb(123456789) == "117.74 MB"
    assert convert_bytes_to_gmbkb(1) == "1 bytes"
