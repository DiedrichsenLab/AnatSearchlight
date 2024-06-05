"""
Searchlight familiy class for creating searchlight structures and running searchlight analyses
within the defined structures.

created on 2024-05-16

Author: Diedrichsenlab
"""

import numpy as np
import pandas as pd
import nibabel as nb
import nilearn as nl
import nitools as nt
import h5py # For IO with HDF5 files


def load(fname):
    """Loads a searchlight definition from an HDF5 file

    Args:
        fname (str): Filename

    Returns:
        Searchlight: Searchlight object
    """
    with h5py.File(fname, 'r') as hf:
        a = hf.get('classname')
        S = Searchlight(hf.get('affine'),hf.get('shape'))
        S.load(hf)
        hf.close()
    return S

class Searchlight:
    """Base class for searchlight analyses. This class implements the basic behaviors of a searchlight.
    """
    def __init__(self,affine,shape):
        pass

    def define(self):
        """ Computes the voxel list for a searchlight. Needs to be implemented by the child class."""
        pass

    def load(self,hf):
        """Loads all obligatory fields from the h5 file

        Args:
            hf (h5py.file): h5 file object (opened in read mode)
        """


    def save(self,fname):
        """Saves the defined searchlight definition to hd5 file
        Args:
            fname (str): Filename
        """
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('classname' = self.classname)
            hf.create_dataset('structure', data=self.structure)
            hf.create_dataset('affine', data=self.affine)
            hf.create_dataset('shape', data=self.shape)
            hf.create_dataset('center_indx', data=self.center_indx)
            hf.create_dataset('voxlist', data=self.voxlist)
            hf.create_dataset('voxmin', data=self.voxmin)
            hf.create_dataset('voxmax', data=self.voxmax)
            hf.create_dataset('maxdist', data=self.maxdist)
            hf.close()

    def run(self):
        pass

class SearchlightVolume(Searchlight):
    """ Anatomically informed searchlights for 3d volumes, given an ROI image.
    Voxels we picked from the mask_img, if an extra mask_image is provided.
    """
    def __init__(self,structure='none'):
        """Constructor for SearchlightVolume with either fixed radius or fixed number of voxels.
        If both a set to a value, nvoxels is used up to a maximum of the radius.

        Args:
            roi_img (filename of NiftiImage): ROI image (or file name) to define output and searchlight locations
            mask_img (filename or NiftiImage): Mask image to define input space, By default the ROI image is used. Defaults to None.
            radius (float): Maximum searchlight radius - set to None if you want a fixed number of voxels
            nvoxels (int): Number of voxels in the searchlight.
        """
        self.classname ='SearchlightVolume'
        self.structure = structure

    def define(self,roi_img,mask_img=None,radius=5,nvoxels=None):
        """ Computes the voxel list for a Volume-based searchlight."""
        if isinstance(roi_img,str):
            self.roi_img = nb.load(roi_img)
        elif isinstance(roi_img,nb.Nifti1Image):
            self.roi_img = roi_img
        else:
            raise ValueError("roi_img must be a filename or Nifti1Image")

        if mask_img is not None:
            if isinstance(mask_img,str):
                self.mask_img = nb.load(mask_img)
            elif isinstance(mask_img,nb.Nifti1Image):
                self.mask_img = mask_img
            else:
                raise ValueError("mask_img must be a filename or Nifti1Image")
        else:
            self.mask_img = None
        self.radius = radius
        self.nvoxels = nvoxels


        i,j,k = np.where(self.roi_img.get_fdata())
        self.center_indx = np.array([i,j,k]).astype('int16')
        self.n_cent = len(i)
        center_coords = nt.affine_transform_mat(self.center_indx,self.affine)

        if self.mask_img is not None:
            i,j,k = np.where(self.mask_img.get_fdata())
            voxel_indx = np.array([i,j,k]).astype('int16')
            voxel_coords = nt.affine_transform_mat(voxel_indx,self.affine)
        else:
            voxel_indx = self.center_indx
            voxel_coords = center_coords

        linvoxel_indx = np.ravel_multi_index(voxel_indx,self.shape).astype('int32')

        self.voxlist = []
        self.voxmin = np.zeros((self.n_cent,3),dtype='int16') # Bottom left voxel
        self.voxmax = np.zeros((self.n_cent,3),dtype='int16')
        self.maxdist = np.zeros(self.n_cent)
        self.affine = roi_img.affine # Affine transformation matrix of data and output space
        self.shape = roi_img.shape   # Shape of the data and output space

        for i in range(self.n_cent):
            if np.mod(i,1000)==0:
                print(f"Processing center {i} of {self.n_cent}")
            dist = nt.euclidean_dist_sq(center_coords[:,i],voxel_coords).squeeze()
            if self.nvoxels is None:
                vi=np.where([dist<(self.radius**2)])[0]
            elif self.radius is None:
                vi=np.argsort(dist)[:self.nvoxels]
            else:
                vi = np.where(dist<(self.radius**2))[0]
                if len(vi)>self.nvoxels:
                    vi=np.argsort(dist[vi])[:self.nvoxels]
            self.voxlist.append(linvoxel_indx[vi])
            self.voxmin[i,:] = np.min(voxel_coords[:,vi],axis=1)
            self.voxmax[i,:] = np.max(voxel_coords[:,vi],axis=1)
            self.maxdist[i] = np.max(dist[vi])



class SearchlightSurface(Searchlight):
    def __init__(self):
        pass

class SearchlightSet(Searchlight):
    def __init__(self,list_of_searchlights):
        pass