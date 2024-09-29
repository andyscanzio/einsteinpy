import numpy as np
from astropy import units as u

from einsteinpy import constant, metric
from einsteinpy.coordinates.conversion import (
    BoyerLindquistConversion,
    CartesianConversion,
    SphericalConversion,
)
from einsteinpy.coordinates.utils import v0
from einsteinpy.utils import CoordinateError

_c = constant.c.value


class BaseDifferential:
    def __init__(
        self, e0, e1, e2, e3, u1, u2, u3, system, name_e0, name_e1, name_e2, name_e3, name_u0, name_u1, name_u2, name_u3
    ):
        self.e0 = e0
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self._u0 = None
        self.u1 = u1
        self.u2 = u2
        self.u3 = u3
        self.system = system
        self.name_map = {
            name_e0: "e0",
            name_e1: "e1",
            name_e2: "e2",
            name_e3: "e3",
            name_u0: "u0",
            name_u1: "u1",
            name_u2: "u2",
            name_u3: "u3",
        }
        self.name_list_e = [name_e0, name_e1, name_e2, name_e3]
        self.name_list_u = [
            name_e0,
            name_e1,
            name_e2,
            name_e3,
        ]

    def stringify(self):
        values_e = ", ".join(
            f"{name} = ({getattr(self, name)})" for name in self.name_list_e
        )
        values_u = ", ".join(
            f"{name}: {getattr(self, name)}" for name in self.name_list_u
        )
        return f"{self.system} Coordinates: \n \
            {values_e}\n\
            {values_u}"

    __str__ = stringify
    __repr__ = stringify

    def __getattr__(self, attr):
        if attr == 'u0':
            return self._u0
        if attr in self.name_map:
            return getattr(self, self.name_map[attr])
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )

    def __setattr__(self, attr, value):
        if "name_map" in self.__dict__ and attr in self.name_map:
            object.__setattr__(self, self.name_map[attr], value)
        else:
            object.__setattr__(self, attr, value)

    def position(self):
        """
        Returns Position 4-Vector in SI units

        Returns
        -------
        tuple
            4-Tuple, containing Position 4-Vector in SI units

        """
        return (
            _c * self.e0.si.value,
            self.e1.si.value,
            self.e2.si.value,
            self.e3.si.value,
        )

    @property
    def u0(self):
        """
        Returns the Timelike component of 4-Velocity

        """
        return self._v_t

    @u0.setter
    def u0(self, args):
        """
        Sets the value of the Time-like component of 4-Velocity

        Parameters
        ----------
        args : tuple
            1-tuple containing the ~einsteinpy.metric.* object, \
            in which the coordinates are defined

        Raises
        ------
        CoordinateError
            If ``metric`` object has been instantiated with a coordinate system, \
            other than Cartesian Coordinates.

        """
        g = args[0]
        if self.system != g.coords.system:
            raise CoordinateError(
                f"Metric object has been instantiated with a coordinate system, ( {g.coords.system} )"
                " other than Cartesian Coordinates."
            )

        g_cov_mat = g.metric_covariant(self.position())

        u0 = v0(g_cov_mat, self.u1.si.value, self.u2.si.value, self.u3.si.value)

        self._u0 = u0 * u.m / u.s

    def velocity(self, metric):
        """
        Returns Velocity 4-Vector in SI units

        Parameters
        ----------
        metric : ~einsteinpy.metric.*
            Metric object, in which the coordinates are defined

        Returns
        -------
        tuple
            4-Tuple, containing Velocity 4-Vector in SI units

        """
        # Setting _v_t
        self.u0 = (metric,)

        return (
            self._u0.value,
            self.u1.si.value,
            self.u2.si.value,
            self.u3.si.value,
        )


class CartesianDifferential(BaseDifferential, CartesianConversion):
    """
    Class for defining 3-Velocity & 4-Velocity in Cartesian Coordinates \
    using SI units

    """

    @u.quantity_input(
        e0=u.s, e1=u.m, e2=u.m, e3=u.m, u1=u.m / u.s, u2=u.m / u.s, u3=u.m / u.s
    )
    def __init__(self, e0, e1, e2, e3, u1, u2, u3):
        """
        Constructor

        Parameters
        ----------
        e0 : ~astropy.units.quantity.Quantity
            Time
        e1 : ~astropy.units.quantity.Quantity
            x-Component of 3-Position
        e2 : ~astropy.units.quantity.Quantity
            y-Component of 3-Position
        e3 : ~astropy.units.quantity.Quantity
            z-Component of 3-Position
        u1 : ~astropy.units.quantity.Quantity, optional
            x-Component of 3-Velocity
        u2 : ~astropy.units.quantity.Quantity, optional
            y-Component of 3-Velocity
        u3 : ~astropy.units.quantity.Quantity, optional
            z-Component of 3-Velocity

        """
        CartesianConversion.__init__(
            self,
            e0.si.value,
            e1.si.value,
            e2.si.value,
            e3.si.value,
            u1.si.value,
            u2.si.value,
            u3.si.value,
        )
        BaseDifferential.__init__(
            self, e0, e1, e2, e3, u1, u2, u3, "Cartesian", "t", "x", "y", "z", 'v_t', 'v_x', 'v_y','v_z'
        )

    def spherical_differential(self, **kwargs):
        """
        Converts to Spherical Polar Coordinates

        Parameters
        ----------
        **kwargs : dict
            Keyword Arguments

        Returns
        -------
        ~einsteinpy.coordinates.differential.SphericalDifferential
            Spherical Polar representation of velocity

        """
        e0, e1, e2, e3, u1, u2, u3 = self.convert_spherical()
        return SphericalDifferential(
            e0 * u.s,
            e1 * u.m,
            e2 * u.rad,
            e3 * u.rad,
            u1 * u.m / u.s,
            u2 * u.rad / u.s,
            u3 * u.rad / u.s,
        )

    def bl_differential(self, **kwargs):
        """
        Converts to Boyer-Lindquist Coordinates

        Parameters
        ----------
        **kwargs : dict
            Keyword Arguments
            Expects two arguments, ``M and ``a``, as described below

        Other Parameters
        ----------------
        M : float
            Mass of the gravitating body, \
            around which, spacetime has been defined
        a : float
            Spin Parameter of the gravitating body, \
            around which, spacetime has been defined

        Returns
        -------
        ~einsteinpy.coordinates.differential.BoyerLindquistDifferential
            Boyer-Lindquist representation of velocity

        """
        M, a = kwargs["M"], kwargs["a"]
        e0, e1, e2, e3, u1, u2, u3 = self.convert_bl(M=M, a=a)
        return BoyerLindquistDifferential(
            e0 * u.s,
            e1 * u.m,
            e2 * u.rad,
            e3 * u.rad,
            u1 * u.m / u.s,
            u2 * u.rad / u.s,
            u3 * u.rad / u.s,
        )


class SphericalDifferential(BaseDifferential, SphericalConversion):
    """
    Class for defining 3-Velocity & 4-Velocity in Spherical Polar Coordinates \
    using SI units

    """

    @u.quantity_input(
        e0=u.s,
        e1=u.m,
        e2=u.rad,
        e3=u.rad,
        u1=u.m / u.s,
        u2=u.rad / u.s,
        u3=u.rad / u.s,
    )
    def __init__(self, e0, e1, e2, e3, u1, u2, u3):
        """
        Constructor

        Parameters
        ----------
        e0 : float
            Time
        e1 : float
            r-Component of 3-Position
        e2 : float
            theta-Component of 3-Position
        e3 : float
            phi-Component of 3-Position
        u1 : float, optional
            r-Component of 3-Velocity
        u2 : float, optional
            theta-Component of 3-Velocity
        u3 : float, optional
            phi-Component of 3-Velocity

        """
        SphericalConversion.__init__(
            self,
            e0.si.value,
            e1.si.value,
            e2.si.value,
            e3.si.value,
            u1.si.value,
            u2.si.value,
            u3.si.value,
        )
        BaseDifferential.__init__(
            self, e0, e1, e2, e3, u1, u2, u3, "Spherical", "t", "r", "theta", "phi", 'v_t', 'v_r', 'v_th', 'v_p'
        )

    def cartesian_differential(self, **kwargs):
        """
        Converts to Cartesian Coordinates

        Parameters
        ----------
        **kwargs : dict
            Keyword Arguments

        Returns
        -------
        ~einsteinpy.coordinates.differential.CartesianDifferential
            Cartesian representation of velocity

        """
        e0, e1, e2, e3, u1, u2, u3 = self.convert_cartesian()
        return CartesianDifferential(
            e0 * u.s,
            e1 * u.m,
            e2 * u.m,
            e3 * u.m,
            u1 * u.m / u.s,
            u2 * u.m / u.s,
            u3 * u.m / u.s,
        )

    def bl_differential(self, **kwargs):
        """
        Converts to Boyer-Lindquist coordinates

        Parameters
        ----------
        **kwargs : dict
            Keyword Arguments
            Expects two arguments, ``M and ``a``, as described below

        Other Parameters
        ----------------
        M : float
            Mass of the gravitating body, \
            around which, spacetime has been defined
        a : float
            Spin Parameter of the gravitating body, \
            around which, spacetime has been defined

        Returns
        -------
        ~einsteinpy.coordinates.differential.BoyerLindquistDifferential
            Boyer-Lindquist representation of velocity

        """
        M, a = kwargs["M"], kwargs["a"]
        e0, e1, e2, e3, u1, u2, u3 = self.convert_bl(M=M, a=a)
        return BoyerLindquistDifferential(
            e0 * u.s,
            e1 * u.m,
            e2 * u.rad,
            e3 * u.rad,
            u1 * u.m / u.s,
            u2 * u.rad / u.s,
            u3 * u.rad / u.s,
        )


class BoyerLindquistDifferential(BaseDifferential, BoyerLindquistConversion):
    """
    Class for defining 3-Velocity & 4-Velocity in Boyer-Lindquist Coordinates \
    using SI units

    """

    @u.quantity_input(
        e0=u.s,
        e1=u.m,
        e2=u.rad,
        e3=u.rad,
        u1=u.m / u.s,
        u2=u.rad / u.s,
        u3=u.rad / u.s,
    )
    def __init__(self, e0, e1, e2, e3, u1, u2, u3):
        """
        Constructor.

        Parameters
        ----------
        e0 : float
            Time
        e1 : float
            r-Component of 3-Position
        e2 : float
            theta-Component of 3-Position
        e3 : float
            phi-Component of 3-Position
        u1 : float, optional
            r-Component of 3-Velocity
        u2 : float, optional
            theta-Component of 3-Velocity
        u3 : float, optional
            phi-Component of 3-Velocity

        """
        BoyerLindquistConversion.__init__(
            self,
            e0.si.value,
            e1.si.value,
            e2.si.value,
            e3.si.value,
            u1.si.value,
            u2.si.value,
            u3.si.value,
        )
        BaseDifferential.__init__(
            self, e0, e1, e2, e3, u1, u2, u3, "BoyerLindquist", "t", "r", "theta", "phi", 'v_t', 'v_r', 'v_th', 'v_p'
        )

    def cartesian_differential(self, **kwargs):
        """
        Converts to Cartesian Coordinates

        Parameters
        ----------
        **kwargs : dict
            Keyword Arguments
            Expects two arguments, ``M and ``a``, as described below

        Other Parameters
        ----------------
        M : float
            Mass of the gravitating body, \
            around which, spacetime has been defined
        a : float
            Spin Parameter of the gravitating body, \
            around which, spacetime has been defined

        Returns
        -------
        ~einsteinpy.coordinates.differentia.CartesianDifferential
            Cartesian representation of velocity

        """
        M, a = kwargs["M"], kwargs["a"]
        e0, e1, e2, e3, u1, u2, u3 = self.convert_cartesian(M=M, a=a)
        return CartesianDifferential(
            e0 * u.s,
            e1 * u.m,
            e2 * u.m,
            e3 * u.m,
            u1 * u.m / u.s,
            u2 * u.m / u.s,
            u3 * u.m / u.s,
        )

    def spherical_differential(self, **kwargs):
        """
        Converts to Spherical Polar Coordinates

        Parameters
        ----------
        **kwargs : dict
            Keyword Arguments
            Expects two arguments, ``M and ``a``, as described below

        Other Parameters
        ----------------
        M : float
            Mass of the gravitating body, \
            around which, spacetime has been defined
        a : float
            Spin Parameter of the gravitating body, \
            around which, spacetime has been defined

        Returns
        -------
        ~einsteinpy.coordinates.differentia.SphericalDifferential
            Spherical representation of velocity

        """
        M, a = kwargs["M"], kwargs["a"]
        e0, e1, e2, e3, u1, u2, u3 = self.convert_spherical(M=M, a=a)
        return SphericalDifferential(
            e0 * u.s,
            e1 * u.m,
            e2 * u.rad,
            e3 * u.rad,
            u1 * u.m / u.s,
            u2 * u.rad / u.s,
            u3 * u.rad / u.s,
        )
