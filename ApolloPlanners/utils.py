
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from juggling_apollo.utils import *
from examples.utils import plot_A
from juggling_apollo.settings import g
from ApolloKinematics import utilities
