import searchlight as sl
import numpy as np
import nibabel as nib

def mvpa_function(data):
    pass

def test_vol():
    mySearchlight = sl.SearchlightVolume('cerebellum')
    mySearchlight.define('data/sub-02_desc-cereb_mask.nii',radius=4,nvoxels=100)
    mySearchlight.save('searchlight.h5')
    S = sl.load('searchlight.h5')
    S.run('data.nii', 'output.nii')

if __name__ == '__main__':
    test_vol()