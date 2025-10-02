# AnatSearchlight
This repository includes code to run anatomically informed searchlight for functional MRI data using python.

The `SearchlightVolume` class  defines a searchlight sphere in 3D space, restricted to a specific ROI. For example, you can use it for a search light for each cerebellar voxel, using only voxels from the cerebellum.

The `SearchlightSurface` class  defines a searchlight for a cortical hemisphere. One searchlight for each vertex is computed.

Running a searchlight analysis involves two main steps:

* define: defining the searchlights. The precomputed searchlight can be saved as a .h5 file.
* run: runs the searchlight analysis for arbitrary input data and mvpa-function. The results can be saved as nifti or cifti files.

## Dependencies
The code is written in python 3.9 and uses the following libraries:
- numpy
- nibabel

We are also using the surface class written by Nick Oosterhof, which was originally part of the PyMVPA project.

## Example:

See examples/SearchlightVolume_example.ipynb for an example of how to use the `SearchlightVolume` class.
See examples/SearchlightSurface_example.ipynb for an example of how to use the `SearchlightSurface` class.
