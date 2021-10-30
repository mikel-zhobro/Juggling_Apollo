
import numpy as np




class Link:
    """
    Base Link class.
    Parameters
    ----------
    name: string
        The name of the link
    bounds: tuple
        Optional : The bounds of the link. Defaults to None
    Attributes
    ----------
    has_rotation: bool
        Whether the link provides a rotation
    length: float
        Length of the link
    """

    def __init__(self, name, length, bounds=(None, None), is_final=False):
        self.bounds = bounds
        self.name = name
        self.length = length
        self.axis_length = length
        self.is_final = is_final
        self.has_rotation = False
        self.joint_type = None

    def __repr__(self):
        return "Link name={} bounds={}".format(self.name, self.bounds)

    def get_rotation_axis(self):
        """
        Returns
        -------
        coords:
            coordinates of the rotation axis in the frame of the joint
        """
        # Defaults to None
        raise ValueError("This Link doesn't have a rotation axis")

    def get_link_frame_matrix(self, actuator_parameters: dict):
        """
        Return the frame matrix corresponding to the link, parameterized with theta
        Parameters
        ----------
        actuator_parameters: dict
            Values for the actuator movements
        Note
        ----
        Theta works for rotations, and for other one-dimensional actuators (ex: prismatic joints), even if the name can be misleading
        """
        raise NotImplementedError


class DHLink(Link):
    """Link in Denavit-Hartenberg representation.
    Parameters
    ----------
    name: str
        The name of the link
    bounds: tuple
        Optional : The bounds of the link. Defaults to None
    d: float
        offset along previous z to the common normal
    a: float
        offset along previous   to the common normal
    use_symbolic_matrix: bool
        whether the transformation matrix is stored as Numpy array or as a Sympy symbolic matrix.
    Returns
    -------
    DHLink:
        The link object
    """

    def __init__(self, name, d=0, a=0, bounds=None, use_symbolic_matrix=True):
        Link.__init__(self, use_symbolic_matrix)
        self.d = d
        self.a = a

    def get_link_frame_matrix(self, parameters):
        """ Computes the homogeneous transformation matrix for this link. """
        theta = parameters
        ct = np.cos(theta + self.theta)
        st = np.sin(theta + self.theta)
        ca = np.cos(self.alpha)
        sa = np.sin(self.alpha)

        return np.matrix(((ct, -st * ca, st * sa, self.a * ct),
                          (st, ct * ca, -ct * sa, self.a * st),
                          (0, sa, ca, self.d),
                          (0, 0, 0, 1)))


class OriginLink(Link):
    """The link at the origin of the robot"""
    def __init__(self):
        Link.__init__(self, name="Base link", length=1)
        self.has_rotation = False
        self.has_translation = False
        self.joint_type = "fixed"

    def get_rotation_axis(self):
        return [0, 0, 0, 1]

    def get_link_frame_matrix(self, theta):
        return np.eye(4)