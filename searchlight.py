"""
Searchlight familiy class for creating searchlight structures and running searchlight analyses
within the defined structures.

created on 2024-05-16

Author: Bassel Arafat
"""

import numpy as np
import pandas as pd
import nibabel as nb
import nilearn as nl
import pytorch as pt


class Searchlight:
    """Base class for searchlight analyses. This class is used to define the searchlight,
    save the searchlight structure and run the searchlight analysis on the defined structure."""
    def __init__(self):
        pass
    def define_searchlight(self):
        pass
    def save_searchlight(self):
        pass
    def run_searchlight(self):
        pass

class SearchlightROI(Searchlight):
    def __init__(self):
        pass

class Searchlight(Searchlight):
    def __init__(self):
        pass
class SearchlightSet(Searchlight):
    def __init__(self):
        pass