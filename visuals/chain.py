# coding= utf8
"""
.. module:: chain
This module implements the Chain class.
"""
import numpy as np
import json
import os
from typing import List
import warnings


class Chain:
    """The base Chain class

    Parameters
    ----------
    links: list[ikpy.link.Link]
        List of the links of the chain
    active_links_mask: list
        A list of boolean indicating that whether or not the corresponding link is active
    name: str
        The name of the Chain
    """
    def __init__(self, links, active_links_mask=None, name="chain", **kwargs):
        self.name = name
        self.links = links
        self._length = sum([link.length for link in links])
        # Avoid length of zero in a link
        for (index, link) in enumerate(self.links):
            if link.length == 0:
                link.axis_length = self.links[index - 1].axis_length

        # If the active_links_mask is not given, set it to True for every link
        if active_links_mask is not None:
            if len(active_links_mask) != len(self.links):
                raise ValueError("Your active links mask length of {} is different from the number of your links, which is {}".format(len(active_links_mask), len(self.links)))
            self.active_links_mask = np.array(active_links_mask)

        else:
            self.active_links_mask = np.array([True] * len(links))

        # Always set the last link to True
        if self.active_links_mask[-1] is True:
            warnings.warn("active_link_mask[-1] is True, but it should be set to False. Overriding and setting to False")
            self.active_links_mask[-1] = False

        # Check that none of the active links are fixed
        for link_index, (link_active, link) in enumerate(zip(self.active_links_mask, self.links)):
            if link.joint_type == "fixed" and link_active:
                warnings.warn("Link {} (index: {}) is of type 'fixed' but set as active in the active_links_mask. In practice, this fixed link doesn't provide any transformation so is as it were inactive".format(link.name, link_index))

    def __repr__(self):
        return "Kinematic chain name={} links={} active_links={}".format(self.name, [link.name for link in self.links], self.active_links_mask)

    def __len__(self):
        return len(self.links)

    def forward_kinematics(self, joints: List, full_kinematics=False):
        """Returns the transformation matrix of the forward kinematics

        Parameters
        ----------
        joints: list
            The list of the positions of each joint. Note : Inactive joints must be in the list.
        full_kinematics: bool
            Return the transformation matrices of each joint

        Returns
        -------
        frame_matrix:
            The transformation matrix
        """
        frame_matrix = np.eye(4)

        if full_kinematics:
            frame_matrixes = []

        if len(self.links) != len(joints):
            raise ValueError("Your joints vector length is {} but you have {} links".format(len(joints), len(self.links)))

        for index, (link, joint_parameters) in enumerate(zip(self.links, joints)):
            # Compute iteratively the position
            # NB : Use asarray to avoid old sympy problems
            # FIXME: The casting to array is a loss of time
            frame_matrix = np.dot(frame_matrix, np.asarray(link.get_link_frame_matrix(joint_parameters)))
            if full_kinematics:
                # rotation_axe = np.dot(frame_matrix, link.rotation)
                frame_matrixes.append(frame_matrix)

        # Return the matrix, or matrixes
        if full_kinematics:
            return frame_matrixes
        else:
            return frame_matrix

    def plot(self, joints, ax, target=None, show=False):
        """Plots the Chain using Matplotlib

        Parameters
        ----------
        joints: list
            The list of the positions of each joint
        ax: matplotlib.axes.Axes
            A matplotlib axes
        target: numpy.array
            An optional target
        show: bool
            Display the axe. Defaults to False
        """
        from utils import plot

        if ax is None:
            # If ax is not given, create one
            ax = plot.init_3d_figure()
        plot.plot_chain(self, joints, ax, name=self.name)

        # Plot the goal position
        if target is not None:
            plot.plot_target(target, ax)
        if show:
            plot.show_figure()

    @classmethod
    def concat(cls, chain1, chain2):
        """Concatenate two chains"""
        return cls(links=chain1.links + chain2.links, active_links_mask=chain1.active_links_mask + chain2.active_links_mask)
    
    def active_to_full(self, active_joints, initial_position):
        full_joints = np.array(initial_position, copy=True, dtype=np.float)
        np.place(full_joints, self.active_links_mask, active_joints)
        return full_joints

    def active_from_full(self, joints):
        return np.compress(self.active_links_mask, joints, axis=0)
