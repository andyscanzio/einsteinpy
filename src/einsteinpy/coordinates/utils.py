import numpy as np

from einsteinpy import constant
from einsteinpy.ijit import jit

_c = constant.c.value


@jit
def cartesian_to_spherical(e0, e1, e2, e3, alpha, u1, u2, u3):
    """
    Utility function (jitted) to convert cartesian to spherical.
    This function should eventually result in Coordinate Transformation Graph!

    """
    hyp = np.hypot(e1, e2)
    sph_e1 = np.hypot(hyp, e3)
    sph_e2 = np.arctan2(hyp, e3)
    sph_e3 = np.arctan2(e2, e1)
    n1 = e1**2 + e2**2
    n2 = n1 + e3**2
    sph_u1 = (e1 * u1 + e2 * u2 + e3 * u3) / np.sqrt(n2)
    sph_u2 = (e3 * (e1 * u1 + e2 * u2) - n1 * u3) / (n2 * np.sqrt(n1))
    sph_u3 = -1 * (u1 * e2 - e1 * u2) / n1

    return e0, sph_e1, sph_e2, sph_e3, sph_u1, sph_u2, sph_u3


@jit
def cartesian_to_spherical_novel(e0, e1, e2, e3, alpha):
    """
    Utility function (jitted) to convert cartesian to spherical.
    This function should eventually result in Coordinate Transformation Graph!

    """
    hyp = np.hypot(e1, e2)
    sph_e1 = np.hypot(hyp, e3)
    sph_e2 = np.arctan2(hyp, e3)
    sph_e3 = np.arctan2(e2, e1)

    return e0, sph_e1, sph_e2, sph_e3

@jit
def cartesian_to_bl(e0, e1, e2, e3, alpha, u1, u2, u3):
    """
    Utility function (jitted) to convert cartesian to boyer lindquist.
    This function should eventually result in Coordinate Transformation Graph!

    """
    w = (e1**2 + e2**2 + e3**2) - (alpha**2)
    bl_e0 = np.sqrt(0.5 * (w + np.sqrt((w**2) + (4 * (alpha**2) * (e3**2)))))
    bl_e1 = np.arccos(e3 / bl_e0)
    bl_e2 = np.arctan2(e2, e1)
    dw_dt = 2 * (e1 * u1 + e2 * u2 + e3 * u3)
    bl_u1 = (1 / (2 * bl_e0)) * (
        (dw_dt / 2)
        + (
            (w * dw_dt + 4 * (alpha**2) * e3 * u3)
            / (2 * np.sqrt((w**2) + (4 * (alpha**2) * (e3**2))))
        )
    )
    bl_u2 = (-1 / np.sqrt(1 - np.square(e3 / bl_e0))) * ((u3 * bl_e0 - bl_u1 * e3) / (bl_e0**2))
    bl_u3 = (1 / (1 + np.square(e2 / e1))) * ((u2 * e1 - u1 * e2) / (e1**2))

    return e0, bl_e0, bl_e1, bl_e2, bl_u1, bl_u2, bl_u3


@jit
def cartesian_to_bl_novel(e0, e1, e2, e3, alpha):
    """
    Utility function (jitted) to convert cartesian to boyer lindquist.
    This function should eventually result in Coordinate Transformation Graph!

    """
    w = (e1**2 + e2**2 + e3**2) - (alpha**2)
    bl_e1 = np.sqrt(0.5 * (w + np.sqrt((w**2) + (4 * (alpha**2) * (e3**2)))))
    bl_e2 = np.arccos(e3 / bl_e1)
    bl_e3 = np.arctan2(e2, e1)

    return e0, bl_e1, bl_e2, bl_e3


@jit
def spherical_to_cartesian(e0, e1, e2, e3, alpha, u1, u2, u3):
    """
    Utility function (jitted) to convert spherical to cartesian.
    This function should eventually result in Coordinate Transformation Graph!

    """
    car_e1 = e1 * np.cos(e3) * np.sin(e2)
    car_e2 = e1 * np.sin(e3) * np.sin(e2)
    car_e3 = e1 * np.cos(e2)
    car_u1 = (
        np.sin(e2) * np.cos(e3) * u1
        - e1 * np.sin(e2) * np.sin(e3) * u3
        + e1 * np.cos(e2) * np.cos(e3) * u2
    )
    car_u2 = (
        np.sin(e2) * np.sin(e3) * u1
        + e1 * np.cos(e2) * np.sin(e3) * u2
        + e1 * np.sin(e2) * np.cos(e3) * u3
    )
    car_u3 = np.cos(e2) * u1 - e1 * np.sin(e2) * u2

    return e0, car_e1, car_e2, car_e3, car_u1, car_u2, car_u3


@jit
def spherical_to_cartesian_novel(e0, e1, e2, e3, alpha):
    """
    Utility function (jitted) to convert spherical to cartesian.
    This function should eventually result in Coordinate Transformation Graph!

    """
    car_e1 = e1 * np.cos(e3) * np.sin(e2)
    car_e2 = e1 * np.sin(e3) * np.sin(e2)
    car_e3 = e1 * np.cos(e2)

    return e0, car_e1, car_e2, car_e3


@jit
def bl_to_cartesian(e0, e1, e2, e3, alpha, u1, u2, u3):
    """
    Utility function (jitted) to convert bl to cartesian.
    This function should eventually result in Coordinate Transformation Graph!

    """
    xa = np.sqrt(e1**2 + alpha**2)
    sin_norm = xa * np.sin(e2)
    car_e1 = sin_norm * np.cos(e3)
    car_e2 = sin_norm * np.sin(e3)
    car_e3 = e1 * np.cos(e2)
    car_u1 = (
        (e1 * u1 * np.sin(e2) * np.cos(e3) / xa)
        + (xa * np.cos(e2) * np.cos(e3) * u2)
        - (xa * np.sin(e2) * np.sin(e3) * u3)
    )
    car_u2 = (
        (e1 * u1 * np.sin(e2) * np.sin(e3) / xa)
        + (xa * np.cos(e2) * np.sin(e3) * u2)
        + (xa * np.sin(e2) * np.cos(e3) * u3)
    )
    car_u3 = (u1 * np.cos(e2)) - (e1 * np.sin(e2) * u2)

    return e0, car_e1, car_e2, car_e3, car_u1, car_u2, car_u3


@jit
def bl_to_cartesian_novel(e0, e1, e2, e3, alpha):
    """
    Utility function (jitted) to convert bl to cartesian.
    This function should eventually result in Coordinate Transformation Graph!

    """
    xa = np.sqrt(e1**2 + alpha**2)
    sin_norm = xa * np.sin(e2)
    car_e1 = sin_norm * np.cos(e3)
    car_e2 = sin_norm * np.sin(e3)
    car_e3 = e1 * np.cos(e2)

    return e0, car_e1, car_e2, car_e3

conversion_map = {('Cartesian', 'Spherical'):(cartesian_to_spherical, cartesian_to_spherical_novel),
                  ('Cartesian', 'BoyerLindquist'):(cartesian_to_bl, cartesian_to_bl_novel),
                  ('Spherical', 'Cartesian'):(spherical_to_cartesian, spherical_to_cartesian_novel),
                  ('BoyerLindquist', 'Cartesian'):(bl_to_cartesian, bl_to_cartesian_novel),
                  }

def convert_fast(
    from_system, to_system, e0, e1, e2, e3, alpha, u1=None, u2=None, u3=None, velocities_provided=False
):
    convert, convert_novel = conversion_map.get((from_system, to_system))
    if velocities_provided:
        return convert(e0, e1, e2, e3, alpha, u1, u2, u3)
    return convert_novel(e0, e1, e2, e3, alpha)


def lorentz_factor(u1, u2, u3):
    """
    Returns the Lorentz Factor, ``gamma``

    Parameters
    ----------
    v1 : float
        First component of 3-Velocity
    v2 : float
        Second component of 3-Velocity
    v3 : float
        Third component of 3-Velocity

    Returns
    -------
    gamma : float
        Lorentz Factor

    """
    u_vec = np.array([u1, u2, u3])
    u_norm2 = u_vec.dot(u_vec)
    gamma = 1 / np.sqrt(1 - u_norm2 / _c**2)

    return gamma


@jit
def v0(g_cov_mat, u1, u2, u3):
    """
    Utility function to return Timelike component (v0) of 4-Velocity
    Assumes a (+, -, -, -) Metric Signature

    Parameters
    ----------
    g_cov_mat : ~numpy.ndarray
        Matrix, containing Covariant Metric \
        Tensor values, in same coordinates as ``v_vec``
        Numpy array of shape (4,4)
    v1 : float
        First component of 3-Velocity
    v2 : float
        Second component of 3-Velocity
    v3 : float
        Third component of 3-Velocity
    Returns
    -------
    float
        Timelike component of 4-Velocity

    """
    g = g_cov_mat
    # Factor to add to coefficient, C
    fac = -1 * _c**2
    # Defining coefficients for quadratic equation
    A = g[0, 0]
    B = 2 * (g[0, 1] * u1 + g[0, 2] * u2 + g[0, 3] * u3)
    C = (
        (g[1, 1] * u1**2 + g[2, 2] * u2**2 + g[3, 3] * u3**2)
        + 2 * u1 * (g[1, 2] * u2 + g[1, 3] * u3)
        + 2 * u2 * g[2, 3] * u3
        + fac
    )
    D = (B**2) - (4 * A * C)

    u0 = (-B + np.sqrt(D)) / (2 * A)

    return u0
