import searchlight as sl
import numpy as np
import nibabel as nib

def mvpa_function(data):
    return np.mean(data,axis=1)

def test_define_searchlight():
    mySearchlight = sl.SearchlightVolume('cerebellum')
    mySearchlight.define('data/sub-02_desc-cereb_mask.nii',radius=4,nvoxels=100)
    mySearchlight.save('data/searchlight.h5')

def test_run_searchlight():
    datafiles = [f"data/sub-02_ses-s1_run-01_reg-{s:02d}_beta.nii" for s in range(5)]
    S= sl.load('data/searchlight.h5')
    results = S.run(datafiles, mvpa_function, )
    S.save_results(results,'data/output.nii')



if __name__ == '__main__':
    test_define_searchlight()
    #test_run_searchlight()
    pass