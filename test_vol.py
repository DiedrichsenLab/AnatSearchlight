import searchlight as sl
import numpy as np
import nibabel as nib

def mvpa_function(data):
    pass

def test_vol():
    mySearchlight = sl.SearchlightVolume('mask.nii',radius=4,nvoxels=100)
    mySearchlight.define()
    mySearchlight.save('searchlight.h5')
    mySearchlight.run('data.nii', 'output.nii')

if __name__ == '__main__':
    test_vol()