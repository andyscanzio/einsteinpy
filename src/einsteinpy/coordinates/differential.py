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
    def __init__(self, e0, e1, e2, e3, u1, u2, u3, system, name_e0, name_e1, name_e2, name_e3):
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
            name_e0: 'e0',
            name_e1: 'e1',
            name_e2: 'e2',
            name_e3: 'e3',
            f'v_{name_e0}': 'u0',
            f'v_{name_e1}': 'u1',
            f'v_{name_e2}': 'u2',
            f'v_{name_e3}': 'u3',
        }
        self.name_list_e = [name_e0, name_e1, name_e2, name_e3]
        self.name_list_u = [f'v_{name_e0}', f'v_{name_e1}', f'v_{name_e2}', f'v_{name_e3}']


    def stringify(self):
        values_e = ", ".join(
            f"{name} = ({self.name_map.get(name)})" for name in self.name_list_e
        )
        values_u = ", ".join(
            f"{name}: {self.name_map.get(name)}" for name in self.name_list_u
        )
        return f"{self.system} Coordinates: \n \
            {values_e}\n\
            {values_u}"

    __str__ = stringify
    __repr__ = stringify


    def __getattr__(self, attr):
        if attr in self.name_map:
            return getattr(self, self.name_map[attr])
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
    
    def __setattr__(self, attr, value):
        if 'name_map' in self.__dict__ and attr in self.name_map:
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
        return (_c * self.e0.si.value, self.e1.si.value, self.e2.si.value, self.e3.si.value)

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
        t : ~astropy.units.quantity.Quantity
            Time
        x : ~astropy.units.quantity.Quantity
            x-Component of 3-Position
        y : ~astropy.units.quantity.Quantity
            y-Component of 3-Position
        z : ~astropy.units.quantity.Quantity
            z-Component of 3-Position
        v_x : ~astropy.units.quantity.Quantity, optional
            x-Component of 3-Velocity
        v_y : ~astropy.units.quantity.Quantity, optional
            y-Component of 3-Velocity
        v_z : ~astropy.units.quantity.Quantity, optional
            z-Component of 3-Velocity

        """
        CartesianConversion.__init__(
            self, e0.si.value, e1.si.value, e2.si.value, e3.si.value, u1.si.value, u2.si.value, u3.si.value
        )
        BaseDifferential.__init__(self, e0, e1, e2, e3, u1, u2, u3, 'Cartesian', 't', 'x', 'y', 'z')

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
        t, r, theta, phi, v_r, v_th, v_p = self.convert_spherical()
        return SphericalDifferential(
            t * u.s,
            r * u.m,
            theta * u.rad,
            phi * u.rad,
            v_r * u.m / u.s,
            v_th * u.rad / u.s,
            v_p * u.rad / u.s,
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
        t, r, theta, phi, v_r, v_th, v_p = self.convert_bl(M=M, a=a)
        return BoyerLindquistDifferential(
            t * u.s,
            r * u.m,
            theta * u.rad,
            phi * u.rad,
            v_r * u.m / u.s,
            v_th * u.rad / u.s,
            v_p * u.rad / u.s,
        )


class SphericalDifferential(SphericalConversion):
    """
    Class for defining 3-Velocity & 4-Velocity in Spherical Polar Coordinates \
    using SI units

    """

    @u.quantity_input(
        t=u.s,
        r=u.m,
        theta=u.rad,
        phi=u.rad,
        v_r=u.m / u.s,
        v_th=u.rad / u.s,
        v_p=u.rad / u.s,
    )
    def __init__(self, t, r, theta, phi, v_r, v_th, v_p):
        """
        Constructor

        Parameters
        ----------
        t : float
            Time
        r : float
            r-Component of 3-Position
        theta : float
            theta-Component of 3-Position
        phi : float
            phi-Component of 3-Position
        v_r : float, optional
            r-Component of 3-Velocity
        v_th : float, optional
            theta-Component of 3-Velocity
        v_p : float, optional
            phi-Component of 3-Velocity

        """
        super().__init__(
            t.si.value,
            r.si.value,
            theta.si.value,
            phi.si.value,
            v_r.si.value,
            v_th.si.value,
            v_p.si.value,
        )
        self.t = t
        self.r = r
        self.theta = theta
        self.phi = phi
        self._v_t = None
        self.v_r = v_r
        self.v_th = v_th
        self.v_p = v_p
        self.system = "Spherical"

    def __str__(self):
        return f"Spherical Polar Coordinates: \n\
            t = ({self.t}), r = ({self.r}), theta = ({self.theta}), phi = ({self.phi})\n\
            v_t: {self.v_t}, v_r: {self.v_r}, v_th: {self.v_th}, v_p: {self.v_p}"

    def __repr__(self):
        return f"Spherical Polar Coordinates: \n\
            t = ({self.t}), r = ({self.r}), theta = ({self.theta}), phi = ({self.phi})\n\
            v_t: {self.v_t}, v_r: {self.v_r}, v_th: {self.v_th}, v_p: {self.v_p}"

    def position(self):
        """
        Returns Position 4-Vector in SI units

        Returns
        -------
        tuple
            4-Tuple, containing Position 4-Vector in SI units

        """
        return (
            _c * self.t.si.value,
            self.r.si.value,
            self.theta.si.value,
            self.phi.si.value,
        )

    @property
    def v_t(self):
        """
        Returns the Timelike component of 4-Velocity

        """
        return self._v_t

    @v_t.setter
    def v_t(self, args):
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
            other than Sperical Polar Coordinates.

        """
        g = args[0]
        if self.system != g.coords.system:
            raise CoordinateError(
                f"Metric object has been instantiated with a coordinate system, ( {g.coords.system} )"
                " other than Spherical Polar Coordinates."
            )

        g_cov_mat = g.metric_covariant(self.position())

        v_t = v0(g_cov_mat, self.v_r.si.value, self.v_th.si.value, self.v_p.si.value)

        self._v_t = v_t * u.m / u.s

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
        self.v_t = (metric,)

        return (
            self._v_t.value,
            self.v_r.si.value,
            self.v_th.si.value,
            self.v_p.si.value,
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
        t, x, y, z, v_x, v_y, v_z = self.convert_cartesian()
        return CartesianDifferential(
            t * u.s,
            x * u.m,
            y * u.m,
            z * u.m,
            v_x * u.m / u.s,
            v_y * u.m / u.s,
            v_z * u.m / u.s,
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
        t, r, theta, phi, v_r, v_th, v_p = self.convert_bl(M=M, a=a)
        return BoyerLindquistDifferential(
            t * u.s,
            r * u.m,
            theta * u.rad,
            phi * u.rad,
            v_r * u.m / u.s,
            v_th * u.rad / u.s,
            v_p * u.rad / u.s,
        )


class BoyerLindquistDifferential(BoyerLindquistConversion):
    """
    Class for defining 3-Velocity & 4-Velocity in Boyer-Lindquist Coordinates \
    using SI units

    """

    @u.quantity_input(
        t=u.s,
        r=u.m,
        theta=u.rad,
        phi=u.rad,
        v_r=u.m / u.s,
        v_th=u.rad / u.s,
        v_p=u.rad / u.s,
    )
    def __init__(self, t, r, theta, phi, v_r, v_th, v_p):
        """
        Constructor.

        Parameters
        ----------
        t : float
            Time
        r : float
            r-Component of 3-Position
        theta : float
            theta-Component of 3-Position
        phi : float
            phi-Component of 3-Position
        v_r : float, optional
            r-Component of 3-Velocity
        v_th : float, optional
            theta-Component of 3-Velocity
        v_p : float, optional
            phi-Component of 3-Velocity

        """
        super().__init__(
            t.si.value,
            r.si.value,
            theta.si.value,
            phi.si.value,
            v_r.si.value,
            v_th.si.value,
            v_p.si.value,
        )
        self.t = t
        self.r = r
        self.theta = theta
        self.phi = phi
        self._v_t = None
        self.v_r = v_r
        self.v_th = v_th
        self.v_p = v_p
        self.system = "BoyerLindquist"

    def __str__(self):
        return f"Boyer-Lindquist Coordinates: \n\
            t = ({self.t}), r = ({self.r}), theta = ({self.theta}), phi = ({self.phi})\n\
            v_t: {self.v_t}, v_r: {self.v_r}, v_th: {self.v_th}, v_p: {self.v_p}"

    def __repr__(self):
        return f"Boyer-Lindquist Coordinates: \n\
            t = ({self.t}), r = ({self.r}), theta = ({self.theta}), phi = ({self.phi})\n\
            v_t: {self.v_t}, v_r: {self.v_r}, v_th: {self.v_th}, v_p: {self.v_p}"

    def position(self):
        """
        Returns Position 4-Vector in SI units

        Returns
        -------
        tuple
            4-Tuple, containing Position 4-Vector in SI units

        """
        return (
            _c * self.t.si.value,
            self.r.si.value,
            self.theta.si.value,
            self.phi.si.value,
        )

    @property
    def v_t(self):
        """
        Returns the Timelike component of 4-Velocity

        """
        return self._v_t

    @v_t.setter
    def v_t(self, args):
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
            other than Boyer-Lindquist Coordinates.

        """
        g = args[0]
        if self.system != g.coords.system:
            raise CoordinateError(
                "Metric object has been instantiated with a coordinate system, ( {g.coords.system} )"
                " other than Boyer-Lindquist Coordinates."
            )

        g_cov_mat = g.metric_covariant(self.position())

        v_t = v0(g_cov_mat, self.v_r.si.value, self.v_th.si.value, self.v_p.si.value)

        self._v_t = v_t * u.m / u.s

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
        self.v_t = (metric,)

        return (
            self._v_t.value,
            self.v_r.si.value,
            self.v_th.si.value,
            self.v_p.si.value,
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
        t, x, y, z, v_x, v_y, v_z = self.convert_cartesian(M=M, a=a)
        return CartesianDifferential(
            t * u.s,
            x * u.m,
            y * u.m,
            z * u.m,
            v_x * u.m / u.s,
            v_y * u.m / u.s,
            v_z * u.m / u.s,
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
        t, r, theta, phi, v_r, v_th, v_p = self.convert_spherical(M=M, a=a)
        return SphericalDifferential(
            t * u.s,
            r * u.m,
            theta * u.rad,
            phi * u.rad,
            v_r * u.m / u.s,
            v_th * u.rad / u.s,
            v_p * u.rad / u.s,
        )
