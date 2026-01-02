Getting Started
===============

This guide introduces the typical workflow for running a searchlight analysis.

Core Concepts
-------------

A *searchlight* is defined by:

* A set of **centers** (voxels or surface vertices)
* A set of **input voxels** associated with each center,  defined by a radius or number of voxels.

AnatSearchlight separates the workflow into two stages:

1. **Definition stage**
   Computes and stores the voxel lists for each searchlight center.
2. **Execution stage**
   Applies a user-defined analysis function to data sampled from each searchlight. This function can be defined or imported from other toolboxes (e.g., rsatoolbox).

This separation allows the same searchlight file to be reused for different functions.

Basic Workflow
--------------

1. Create a searchlight object
2. Define the searchlight geometry
3. Save the searchlight (optional)
4. Run an analysis
5. Map results back to image space

Example (Volume-based)
----------------------

.. code-block:: python
    
    from AnatSearchlight import SearchlightVolume
    import numpy as np

    S = SearchlightVolume(structure='cerebellum')

    S.define(
        roi_img='roi.nii.gz',
        maxradius=10,
        maxvoxels=np.inf
    )

    results = S.run(
        inputfiles=[
            'sub-01_beta-001.nii.gz',
            'sub-01_beta-002.nii.gz',
            'sub-01_beta-003.nii.gz'
        ],
        mvpa_function=my_mvpa_function
    )

    img = S.data_to_nifti(results)