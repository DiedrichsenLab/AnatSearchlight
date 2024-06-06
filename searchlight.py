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
        classname = hf.get('classname').asstr()[()]
        structure = hf.get('structure').asstr()[()]
        if classname == 'SearchlightVolume':
            S = SearchlightVolume(structure)
        elif classname == 'SearchlightSurface':
            S = SearchlightSurface(structure)
        else:
            raise ValueError(f"Classname {classname} not recognized")
        S.load(hf)
        hf.close()
    return S

class Searchlight:
    """Base class for searchlight analyses. This class implements the basic behaviors of a searchlight.
    """

    def define(self):
        """ Computes the voxel list for a searchlight. Needs to be implemented by the child class."""
        pass

    def load(self,hf):
        """Loads all obligatory fields from the h5 file

        Args:
            hf (h5py.file): h5 file object (opened in read mode)
        """
        self.affine= np.array(hf.get('affine'))
        self.shape = np.array(hf.get('shape'))
        self.center_indx = np.array(hf.get('center_indx'))
        self.voxel_indx = np.array(hf.get('voxel_indx'))
        self.voxlist = []
        for i in range(len(hf.get('voxlist'))):
            self.voxlist.append(np.array(hf.get(f'voxlist/voxlist_{i}')))
        self.voxmin = np.array(hf.get('voxmin'))
        self.voxmax = np.array(hf.get('voxmax'))
        self.maxdist = np.array(hf.get('maxdist'))
        self.n_cent = self.center_indx.shape[1]

    def save(self,fname):
        """Saves the defined searchlight definition to hd5 file
        Args:
            fname (str): Filename
        """
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('classname',data = self.classname)
            hf.create_dataset('structure', data=self.structure)
            hf.create_dataset('affine', data=self.affine)
            hf.create_dataset('shape', data=self.shape)
            hf.create_dataset('center_indx', data=self.center_indx)
            hf.create_dataset('voxel_indx', data=self.voxel_indx)
            grp = hf.create_group('voxlist')
            for i in range(len(self.voxlist)):
                grp.create_dataset(f'voxlist_{i}', data=self.voxlist[i])
            hf.create_dataset('voxmin', data=self.voxmin)
            hf.create_dataset('voxmax', data=self.voxmax)
            hf.create_dataset('maxdist', data=self.maxdist)
            hf.close()

    def run(self,inputfiles,mvpa_function,function_args={}):
        """ Conducts a searchlight analysis for all the searchlights defined in vox_list."""
        # Load the header from all the input files
        input_vols = [nb.load(f) for f in inputfiles]

        # Sample only the voxels that we require
        data = np.empty((len(input_vols),self.voxel_indx.shape[1]))
        for i,v in enumerate(input_vols):
            for j in range(self.voxel_indx.shape[1]):
                data[i,j] = v.dataobj[self.voxel_indx[0,j],self.voxel_indx[1,j],self.voxel_indx[2,j]]

        # Call the mvpa function
        results = []
        for i in range(self.center_indx.shape[1]):
            local_data = data[:,self.voxlist[i]]
            results.append(mvpa_function(local_data,**function_args))
        results = np.array(results)
        return results

    def save_results(self,results,outfilename):
        """Save the results from a single searchlight to file
        Args:
            results (np.array): Results of the searchlight analysis
            outfilenames (str): Filename
        """
        if results.shape[0] != self.n_cent:
            raise ValueError(f"Results must have the same number of elements as the number of searchlights ({self.n_cent})")

        # Put the data into a 4d-matrix
        shp = np.append(self.shape,results.shape[1])
        results_img = np.zeros(shp,dtype='float32')
        results_img[self.center_indx[0,:],self.center_indx[1,:],self.center_indx[2,:]] = results

        # First deal with a single 4d-nifti as an output
        if outfilename is str:
            img = nb.Nifti1Image(results_img,self.affine)
            img.to_filename(outfilename)
        else:
            if results.shape[1] != len(outfilename):
                raise ValueError(f"Results must have the same number of elements as the number of output files ({len(outfilename)})")
            for i,outn in enumerate(outfilename):
                img= nb.Nifti1Image(results_img[:,:,:,i],self.affine)
                img.to_filename(outn)



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
        self.affine = self.roi_img.affine # Affine transformation matrix of data and output space
        self.shape = self.roi_img.shape   # Shape of the data and output space


        i,j,k = np.where(self.roi_img.get_fdata())
        self.center_indx = np.array([i,j,k]).astype('int16')
        self.n_cent = len(i)
        center_coords = nt.affine_transform_mat(self.center_indx,self.affine)

        if self.mask_img is not None:
            i,j,k = np.where(self.mask_img.get_fdata())
            self.voxel_indx = np.array([i,j,k]).astype('int16')
            voxel_coords = nt.affine_transform_mat(self.voxel_indx,self.affine)
        else:
            self.voxel_indx = self.center_indx
            voxel_coords = center_coords

        self.voxlist = []
        self.voxmin = np.zeros((self.n_cent,3),dtype='int16') # Bottom left voxel
        self.voxmax = np.zeros((self.n_cent,3),dtype='int16')
        self.maxdist = np.zeros(self.n_cent)

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
            self.voxlist.append(vi.astype('uint32'))
            self.voxmin[i,:] = np.min(self.voxel_indx[:,vi],axis=1)
            self.voxmax[i,:] = np.max(self.voxel_indx[:,vi],axis=1)
            self.maxdist[i] = np.max(dist[vi])

class SearchlightSurface(Searchlight):
    def __init__(self):
        pass

class SearchlightSet():
    def __init__(self,list_of_searchlights):
        pass
