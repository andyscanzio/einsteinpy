import numpy as np
from astropy import units as u

from einsteinpy import constant
from einsteinpy.coordinates.conversion import (
    BoyerLindquistConversion,
    CartesianConversion,
    SphericalConversion,
)

_c = constant.c.value


class Coordinates:
    def __init__(self, e0, e1, e2, e3, system, name_e0, name_e1, name_e2, name_e3):
        self.e0 = e0
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.system = system
        self.name_map = {name_e0 : self.e0, name_e1: self.e1, name_e2: self.e2, name_e3: self.e3}
        self.name_list = [name_e0, name_e1, name_e2, name_e3]

    def stringify(self):
        values = ', '.join(f"{name} = ({value})" for name, value in self.name_map.items())
        return f"{self.system} Coordinates: \n \
            {values}"
            
    __str__ = stringify
    __repr__ = stringify
    
    def __getitem__(self, item):
        """
        Method to return coordinates
        Objects are subscriptable with both explicit names of \
        parameters and integer indices

        Parameters
        ----------
        item : str or int
            Name of the parameter or its index
            If ``system`` is provided, while initializing, \
            name of the coordinate is returned

        """
        if isinstance(item, (int, np.integer)):
            return self.name_map[self.name_list[item]]
        return self.name_map[item]
    
    
    def __getattr__(self, name):
        return self.name_map.get(name)
    
    def position(self):
        """
        Returns Position 4-Vector in SI units

        Returns
        -------
        tuple
            4-Tuple, containing Position 4-Vector in SI units

        """
        return (_c * self.e0.si.value, self.e1.si.value, self.e2.si.value, self.e3.si.value)
        
        

class Cartesian(Coordinates, CartesianConversion):
    """
    Class for defining 3-Position & 4-Position in Cartesian Coordinates \
    using SI units

    """

    @u.quantity_input(e0=u.s, e1=u.m, e2=u.m, e3=u.m)
    def __init__(self, e0, e1, e2, e3):
        """
        Constructor

        Parameters
        ----------
        t : float
            Time
        x : float
            x-Component of 3-Position
        y : float
            y-Component of 3-Position
        z : float
            z-Component of 3-Position

        """
        CartesianConversion.__init__(self, e0.si.value, e1.si.value, e2.si.value, e3.si.value)
        Coordinates.__init__(self, e0, e1, e2, e3, "Cartesian", 't', 'x','y','z')

    def to_spherical(self, **kwargs):
        """
        Method for conversion to Spherical Polar Coordinates

        Other Parameters
        ----------------
        **kwargs : dict
            Keyword Arguments

        Returns
        -------
        ~einsteinpy.coordinates.core.Spherical
            Spherical representation of the Cartesian Coordinates

        """
        t, r, theta, phi = self.convert_spherical()

        return Spherical(t * u.s, r * u.m, theta * u.rad, phi * u.rad)

    def to_bl(self, **kwargs):
        """
        Method for conversion to Boyer-Lindquist (BL) Coordinates

        Parameters
        ----------
        **kwargs : dict
            Keyword Arguments
            Expects two arguments, ``M and ``a``, as described below

        Other Parameters
        ----------------
        M : float
            Mass of gravitating body
            Required to calculate ``alpha``, the rotational length \
            parameter
        a : float
            Spin Parameter of gravitating body
            0 <= a <= 1
            Required to calculate ``alpha``, the rotational length \
            parameter

        Returns
        -------
        ~einsteinpy.coordinates.core.BoyerLindquist
            Boyer-Lindquist representation of the Cartesian Coordinates

        """
        M, a = kwargs["M"], kwargs["a"]
        t, r, theta, phi = self.convert_bl(M=M, a=a)

        return BoyerLindquist(t * u.s, r * u.m, theta * u.rad, phi * u.rad)


class Spherical(SphericalConversion):
    """
    Class for defining 3-Position & 4-Position in Spherical Polar Coordinates \
    using SI units

    """

    @u.quantity_input(t=u.s, r=u.m, theta=u.rad, phi=u.rad)
    def __init__(self, t, r, theta, phi):
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

        """
        super().__init__(t.si.value, r.si.value, theta.si.value, phi.si.value)
        self.t = t
        self.r = r
        self.theta = theta
        self.phi = phi
        self.system = "Spherical"
        self._dimension = {
            "t": self.t,
            "r": self.r,
            "theta": self.theta,
            "phi": self.phi,
            "system": self.system,
        }
        self._dimension_order = ("t", "r", "theta", "phi")

    def __str__(self):
        return f"Spherical Polar Coordinates: \n \
            t = ({self.t}), r = ({self.r}), theta = ({self.theta}), phi = ({self.phi})"

    def __repr__(self):
        return f"Spherical Polar Coordinates: \n \
            t = ({self.t}), r = ({self.r}), theta = ({self.theta}), phi = ({self.phi})"

    def __getitem__(self, item):
        """
        Method to return coordinates
        Objects are subscriptable with both explicit names of \
        parameters and integer indices

        Parameters
        ----------
        item : str or int
            Name of the parameter or its index
            If ``system`` is provided, while initializing, \
            name of the coordinate is returned

        """
        if isinstance(item, (int, np.integer)):
            return self._dimension[self._dimension_order[item]]
        return self._dimension[item]

    def position(self):
        """
        Returns Position 4-Vector in SI units

        Returns
        -------
        tuple :
            4-Tuple, containing Position 4-Vector in SI units

        """
        return (
            _c * self.t.si.value,
            self.r.si.value,
            self.theta.si.value,
            self.phi.si.value,
        )

    def to_cartesian(self, **kwargs):
        """
        Method for conversion to Cartesian Coordinates

        Other Parameters
        ----------------
        **kwargs : dict
            Keyword Arguments

        Returns
        -------
        ~einsteinpy.coordinates.core.Cartesian
            Cartesian representation of the Spherical Polar Coordinates

        """
        t, x, y, z = self.convert_cartesian()

        return Cartesian(t * u.s, x * u.m, y * u.m, z * u.m)

    def to_bl(self, **kwargs):
        """
        Method for conversion to Boyer-Lindquist (BL) Coordinates

        Parameters
        ----------
        **kwargs : dict
            Keyword Arguments
            Expects two arguments, ``M and ``a``, as described below

        Other Parameters
        ----------------
        M : float
            Mass of gravitating body
            Required to calculate ``alpha``, the rotational length \
            parameter
        a : float
            Spin Parameter of gravitating body
            0 <= a <= 1
            Required to calculate ``alpha``, the rotational length \
            parameter

        Returns
        -------
        ~einsteinpy.coordinates.core.BoyerLindquist
            Boyer-Lindquist representation of the Spherical \
            Polar Coordinates

        """
        M, a = kwargs["M"], kwargs["a"]
        t, r, theta, phi = self.convert_bl(M=M, a=a)

        return BoyerLindquist(t * u.s, r * u.m, theta * u.rad, phi * u.rad)


class BoyerLindquist(BoyerLindquistConversion):
    """
    Class for defining 3-Position & 4-Position in Boyer-Lindquist Coordinates \
    using SI units

    """

    @u.quantity_input(t=u.s, r=u.m, theta=u.rad, phi=u.rad)
    def __init__(self, t, r, theta, phi):
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

        """
        super().__init__(t.si.value, r.si.value, theta.si.value, phi.si.value)
        self.t = t
        self.r = r
        self.theta = theta
        self.phi = phi
        self.system = "BoyerLindquist"
        self._dimension = {
            "t": self.t,
            "r": self.r,
            "theta": self.theta,
            "phi": self.phi,
            "system": self.system,
        }
        self._dimension_order = ("t", "r", "theta", "phi")

    def __str__(self):
        return f"Boyer-Lindquist Coordinates: \n \
            t = ({self.t}), r = ({self.r}), theta = ({self.theta}), phi = ({self.phi})"

    def __repr__(self):
        return f"Boyer-Lindquist Coordinates: \n \
            t = ({self.t}), r = ({self.r}), theta = ({self.theta}), phi = ({self.phi})"

    def __getitem__(self, item):
        """
        Method to return coordinates
        Objects are subscriptable with both explicit names of \
        parameters and integer indices

        Parameters
        ----------
        item : str or int
            Name of the parameter or its index
            If ``system`` is provided, while initializing, \
            name of the coordinate is returned

        """
        if isinstance(item, (int, np.integer)):
            return self._dimension[self._dimension_order[item]]
        return self._dimension[item]

    def position(self):
        """
        Returns Position 4-Vector in SI units

        Returns
        -------
        tuple :
            4-Tuple, containing Position 4-Vector in SI units

        """
        return (
            _c * self.t.si.value,
            self.r.si.value,
            self.theta.si.value,
            self.phi.si.value,
        )

    def to_cartesian(self, **kwargs):
        """
        Method for conversion to Cartesian Coordinates

        Parameters
        ----------
        **kwargs : dict
            Keyword Arguments
            Expects two arguments, ``M and ``a``, as described below

        Other Parameters
        ----------------
        M : float
            Mass of gravitating body
            Required to calculate ``alpha``, the rotational length \
            parameter
        a : float
            Spin Parameter of gravitating body
            0 <= a <= 1
            Required to calculate ``alpha``, the rotational length \
            parameter

        Returns
        -------
        ~einsteinpy.coordinates.core.Cartesian
            Cartesian representation of the Boyer-Lindquist Coordinates

        """
        M, a = kwargs["M"], kwargs["a"]
        t, x, y, z = self.convert_cartesian(M=M, a=a)

        return Cartesian(t * u.s, x * u.m, y * u.m, z * u.m)

    def to_spherical(self, **kwargs):
        """
        Method for conversion to Spherical Polar Coordinates

        Parameters
        ----------
        **kwargs : dict
            Keyword Arguments
            Expects two arguments, ``M and ``a``, as described below

        Other Parameters
        ----------------
        M : float
            Mass of gravitating body
            Required to calculate ``alpha``, the rotational length \
            parameter
        a : float
            Spin Parameter of gravitating body
            0 <= a <= 1
            Required to calculate ``alpha``, the rotational length \
            parameter

        Returns
        -------
        ~einsteinpy.coordinates.core.Spherical
            Spherical Polar representation of the \
            Boyer-Lindquist Coordinates

        """
        M, a = kwargs["M"], kwargs["a"]
        t, r, theta, phi = self.convert_spherical(M=M, a=a)

        return Spherical(t * u.s, r * u.m, theta * u.rad, phi * u.rad)
