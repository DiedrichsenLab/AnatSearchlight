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
import nitools as nt


class Searchlight:
    """Base class for searchlight analyses. This class implements the basic behaviors of a searchlight.
    """
    def __init__(self,affine,shape):
        self.affine = affine # Affine transformation matrix of data and output space
        self.shape = shape   # Shape of the data and output space
        pass

    def define(self):
        """ Computes the voxel list for a searchlight. Needs to be implemented by the child class."""
        pass

    def save(self,fname):
        """Saves the defined searchlight definition to hd5 file
        Args:
            fname (str): Filename
        """
        pass

    def run(self):
        pass

class SearchlightVolume(Searchlight):
    """ Anatomically informed searchlights for 3d volumes, given an ROI image.
    Voxels we picked from the mask_img, if an extra mask_image is provided.
    """
    def __init__(self,roi_img,mask_img=None,radius=5,nvoxels=None):
        """Constructor for SearchlightVolume with either fixed radius or fixed number of voxels.
        If both a set to a value, nvoxels is used up to a maximum of the radius.

        Args:
            roi_img (filename of NiftiImage): ROI image (or file name) to define output and searchlight locations
            mask_img (filename or NiftiImage): Mask image to define input space, By default the ROI image is used. Defaults to None.
            radius (float): Maximum searchlight radius - set to None if you want a fixed number of voxels
            nvoxels (int): Number of voxels in the searchlight.
        """
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
        super().__init__(self.roi_img.affine,self.roi_img.shape)

    def define(self):
        """ Computes the voxel list for a Volume-based searchlight."""
        i,j,k = np.where(self.roi_img.get_fdata())
        self.center_indx = np.array([i,j,k])
        self.n_cent = len(i)
        center_coords = nt.affine_transform_mat(self.center_indx,self.affine)

        if self.mask_img is not None:
            i,j,k = np.where(self.mask_img.get_fdata())
            voxel_indx = np.array([i,j,k]).T
            voxel_coords = nt.affine_transform_mat(voxel_coords_indx,self.affine)
        else:
            voxel_coords = center_coords
        linvoxel_idx = np.ravel_multi_index(voxel_coords.T,self.shape)

        self.voxlist = []
        self.voxmin = np.zeros((self.n_cent,3)) # Bottom left voxel
        self.voxmax = np.zeros((self.n_cent,3))

        for i in range(self.n_cent):
            dist = nt.euclidean_dist_sq(center_coords[:,i],voxel_coords)
            if self.nvoxels is None:
                self.voxels.append(linvoxel_idx[dist<self.radius])
            elif self.radius is None:
                self.voxels.append(linvoxel_idx[np.argsort(dist)[:self.nvoxels]])
            else:
                lv = linvoxel_idx[dist<self.radius]
                self.voxels.append(lv[:min(self.nvoxels,len(lv))])

        pass

        """
        li          = cell(ncent,1);        % linear indices for voxels
        n           = zeros(ncent,1);       % Number of voxels
        rs          = zeros(ncent,1);       % Searchlight radius
        voxmin      = zeros(ncent,1);       % bottom left voxel
        voxmax      = zeros(ncent,1);       % top right voxel


        %% 4. Estimate linear indices, voxmin/voxmax for a sphere centered at each center index
        spm_progress_bar('Init',100);
        for k=1:ncent
            ds = surfing_eucldist(c_centers(:,k),c_voxels);
            if fixedradius
                a       = voxels(:,ds<radius);
                rs(k,1) = radius;
            else
                i       = find(ds<radius(1));
                [dss,j] = sort(ds(i));
                indx    = min(targetvoxelcount,length(i)); % In case there are not enough voxels within maximal radius
                a       = voxels(:,i(j(1:indx)));
                rs(k,1) = dss(indx);
            end;
            n(k,1)          = size(a,2);
            voxmin(k,1:3)   = min(a,[],2)';
            voxmax(k,1:3)   = max(a,[],2)';
            li{k,1}         = surfing_subs2inds(inclMask.dim,a(1:3,:)')';
            n(k,1)          = numel(li{k});

            spm_progress_bar('set',(k/ncent)*100);
        end;


        %% 5. Setting output
        L.LI        = li;
        L.voxmin    = voxmin;
        L.voxmax    = voxmax;
        L.voxel     = centeridxs;


        %% 6. Cleanung & writing out exclusion mask
        Vin(1) = Mask;
        Vin(2) = inclMask;
        Vo     = Vin(1);
        Vo.fname =  'exclMask.nii';
        % exclMask        = spm_imcalc({Mask.fname; inclMask.fname},'exclMask.nii','((i1>0)-(i2>0)>0)');
        exclMask        = spm_imcalc(Vin,Vo,'((i1>0)-(i2>0)>0)');
        exclMask.data   = spm_read_vols(exclMask);
        try
            delete('inclMask.nii');
        catch
        end;
        """


class SearchlightSurface(Searchlight):
    def __init__(self):
        pass

class SearchlightSet(Searchlight):
    def __init__(self,list_of_searchlights):
        pass