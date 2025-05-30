import AnatSearchlight.searchlight as sl
import numpy as np
import nibabel as nib

def mvpa_mean_function(data):
    # Example of MVPA function that returns a scalar as an output argument
    return np.mean(data)

def mvpa_multi_function(data):
    # Example of MVPA function that returns a vector as an output argument
    return np.mean(data,axis=1)

def test_define_searchlight():
    mySearchlight = sl.SearchlightSurface('left_cortex')
    surf = ['examples/sub-02_space-32k_hemi-L_pial.surf.gii',
            'examples/sub-02_space-32k_hemi-L_white.surf.gii']
    voxel_mask = 'examples/sub-02_ses-s1_mask.nii'
    roi_mask = 'examples/tpl-fs32k_hemi-L_mask.label.gii'
    mySearchlight.define(surf,voxel_mask,roi=roi_mask,maxradius=20,maxvoxels=None)
    mySearchlight.save('examples/searchlight_surf.h5')

def test_run_searchlight_mean():
    datafiles = [f"examples/sub-02_ses-s1_run-01_reg-{s:02d}_beta.nii" for s in range(5)]
    S= sl.load('examples/searchlight_surf.h5')
    results = S.run(datafiles, mvpa_mean_function)
    S.data_to_cifti(results,outfilename='examples/output.dscalar.nii')

def test_run_searchlight_multi():
    datafiles = [f"examples/sub-02_ses-s1_run-01_reg-{s:02d}_beta.nii" for s in range(5)]
    S= sl.load('example_data/searchlight_surf.h5')
    results = S.run(datafiles, mvpa_multi_function)
    S.data_to_cifti(results,outfilename='examples/output2.dscalar.nii')

if __name__ == '__main__':
    test_define_searchlight()
    test_run_searchlight_mean()
    test_run_searchlight_multi()
    pass