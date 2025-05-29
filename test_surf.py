import searchlight as sl
import numpy as np
import nibabel as nib

def mvpa_mean_function(data):
    # Example of MVPA function that returns a scalar as an output argument
    return np.mean(data)

def mvpa_multi_function(data):
    # Example of MVPA function that returns a vector as an output argument
    return np.mean(data,axis=1)

def test_define_searchlight():
    mySearchlight = sl.SearchlightSurface('left_hem')
    surf = ['data/sub-02_space-32k_hemi-L_pial.surf.gii',
            'data/sub-02_space-32k_hemi-L_white.surf.gii']
    mask = 'data/sub-02_ses-s1_mask.nii'
    mySearchlight.define(surf,mask,radius=20,nvoxels=None)
    mySearchlight.save('data/searchlight_surf.h5')

def test_run_searchlight_mean():
    datafiles = [f"data/sub-02_ses-s1_run-01_reg-{s:02d}_beta.nii" for s in range(5)]
    S= sl.load('data/searchlight_surf.h5')
    results = S.run(datafiles, mvpa_mean_function, )
    S.save_results(results,'data/output1.nii')

def test_run_searchlight_multi():
    datafiles = [f"data/sub-02_ses-s1_run-01_reg-{s:02d}_beta.nii" for s in range(5)]
    S= sl.load('data/searchlight.h5')
    results = S.run(datafiles, mvpa_multi_function, )
    S.save_results(results,'data/output2.nii')


if __name__ == '__main__':
    # test_define_searchlight()
    test_run_searchlight_mean()
    pass