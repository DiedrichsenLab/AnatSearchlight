import numpy as np
import nibabel as nb
from scipy.spatial import distance_matrix

class SurfaceSearchlight:
    def __init__(self, Surf, Mask, sphere = [30, 160]):
        self.Surf = Surf
        self.Mask = Mask
        # self.Surf = nb.gifti.GiftiImage.from_filename(Surf).darrays
        # self.Mask = nb.load(Mask_name) # nifti
        self.num_surf = len(Surf)
        self.num_vert = sum([surf['white'].dims[0] for surf in Surf])
        # self.ca = np.concatenate([surf['ca'] for surf in Surf], axis=1)
        self.center_idxs = None
        self.excl_mask = None
        self.radius = None
        self.target_vox_count = None
        self.fixed_radius = None
        self.sphere = sphere

    def define_searchlight_surface(self, linedef=[5, 0, 1], distance_metric='euclidean', progress_step=100, write_mask=0):
        # Checking inputs
        if not isinstance(self.sphere, list) or len(self.sphere) < 1:
            raise ValueError("Sphere parameter must be a list with at least one value")
        if not isinstance(linedef, list) or len(linedef) != 3:
            raise ValueError("Linedef parameter must be a list with exactly three values")
        # Getting sphere definition
        self.radius = self.sphere[0]
        self.fixed_radius = len(self.sphere) == 1
        if not self.fixed_radius:
            self.target_vox_count = self.sphere[1]

        # Generate a single large surface from the component surfaces
        white = np.concatenate([surf['white'] for surf in self.Surf], axis=1)
        pial = np.concatenate([surf['pial'] for surf in self.Surf], axis=1)
        ca = (white + pial) / 2

        # Projection onto the linear voxels
        all_coords = surfing_nodeidxs2coords(white, pial, np.arange(self.num_vert), linedef)
        all_lin_vox_idxs = surfing_coords2linvoxelidxs(all_coords)

        uniq_lin_vox_idxs = self.surfing_uniqueidxsperrow(all_lin_vox_idxs)

        self.center_idxs = np.unique(uniq_lin_vox_idxs[~np.isnan(uniq_lin_vox_idxs) & (uniq_lin_vox_idxs != 0)])

        incl_mask = np.zeros(self.Mask.shape)
        incl_mask[self.center_idxs] = 1
        self.excl_mask = (self.Mask.get_fdata() > 0).astype(int) - (incl_mask > 0).astype(int)

        # Other calculations...
        radius = self.radius
        target_vox_count = self.target_vox_count
        fixed_radius = self.fixed_radius
        ca = self.ca
        num_surf = self.num_surf

        centersubs = surfing_inds2subs(self.Mask.shape, self.center_idxs)
        centercoords = affine_transform(centersubs[:, 0], centersubs[:, 1], centersubs[:, 2], self.Mask.affine)

        lin2sub = surfing_inds2subs(self.Mask.shape, np.arange(1, np.prod(self.Mask.shape) + 1))

        centerorder = np.random.permutation(len(self.center_idxs))

        LI = []
        voxmin = np.empty((len(self.center_idxs), 3))
        voxmax = np.empty((len(self.center_idxs), 3))
        voxel = self.center_idxs.copy()
        n = np.empty((len(self.center_idxs), 1))
        rs = np.empty((len(self.center_idxs), 1))
        ascenter = np.empty((len(self.center_idxs), 1), dtype=bool)

        voxcountsum = 0

        for k in centerorder:
            dist = surfing_eucldist(centercoords[k, :], ca)
            node_idx = np.argmin(dist)
            j = node_idx // self.num_vert
            node_idx %= self.num_vert

            v2 = centercoords[k, :] - self.Surf[j]['white'][:, node_idx]
            v1 = self.Surf[j]['pial'][:, node_idx] - self.Surf[j]['white'][:, node_idx]
            depth = np.dot(v1.T, v2) / np.dot(v1.T, v1)

            radiusk = radius
            if radius == 0:
                radiusk = 5
                radius = 5

            voxcount = 0
            done = False
            while not done:
                node_idxs, dists = self.surfing_circle_roi(self.Surf[j]['ca'], self.Surf[j]['n2f'], node_idx, radiusk, distance_metric)
                linvoxidxs = self.Surf[j]['unqlinvoxidxs'][node_idxs, :]
                n2vk = np.unique(linvoxidxs[linvoxidxs > 0])
                voxcount = len(n2vk)

                if fixed_radius:
                    n[k, 0] = voxcount
                    break
                else:
                    if voxcount < target_vox_count:
                        radiusk *= 1.5
                        if radiusk > 1000:
                            done = True
                    else:
                        n[k, 0] = voxcount
                        rs[k, 0] = dists[-1]
                        break

            LI.append(n2vk.astype(np.uint32))
            ijk = lin2sub[n2vk - 1, :]
            voxmin[k, :] = np.min(ijk, axis=0)
            voxmax[k, :] = np.max(ijk, axis=0)
            ascenter[k] = True
            voxcountsum += voxcount

        return {
            'LI': LI,
            'voxmin': voxmin,
            'voxmax': voxmax,
            'voxel': voxel
        }


def surfing_nodeidxs2coords(cs1, cs2, idxs=None, linedef=None):
    """
    Maps vertex indices from two surfaces to coordinates.

    Parameters:
    cs1 : ndarray
        3xN coordinates for N nodes for the first surface.
    cs2 : ndarray
        3xN coordinates for N nodes for the second surface.
    idxs : ndarray, optional
        Px1 node indices to be used (default: all nodes).
    linedef : list, optional
        1x3 vector [S, MN, MAX] specifying that S steps are taken along the
        lines from nodes on cs1 to cs2. MN and MAX are relative indices, where 0
        corresponds to cs1 and 1 to cs2. Default is [5, 0, 1].

    Returns:
    CS : ndarray
        3xSxP array, where CS[:, I, J] are the coordinates for the I-th step
        on the line connecting the nodes on cs1 and cs2 with index idxs[J].
    """

    # Ensure the input matrices are correctly oriented
    if cs1.shape[0] != 3:
        cs1 = cs1.T
    if cs2.shape[0] != 3:
        cs2 = cs2.T

    if cs1.shape != cs2.shape:
        raise ValueError('Number of vertices in surfaces do not match, or not 3xQ matrices')

    # Use all nodes if idxs is not provided
    if idxs is None or not isinstance(idxs, (np.ndarray, list, tuple)):
        idxs = np.arange(cs1.shape[1])

    # Select nodes
    cs1 = cs1[:, idxs]
    cs2 = cs2[:, idxs]
    ncs = cs1.shape[1]  # number of nodes

    # Default line definition
    default_linedef = [5, 0, 1]
    if linedef is None:
        linedef = default_linedef
    if len(linedef) > 3:
        raise ValueError('Unexpected line definition, should be 1x3')
    linedef = linedef + default_linedef[len(linedef):]

    steps, minpos, maxpos = linedef

    if steps == 1:
        relpos = np.array([(minpos + maxpos) / 2])  # if only one step, take average of start and end position
    else:
        relpos = np.linspace(minpos, maxpos, steps)  # weights for surface 1

    steps_12 = np.tile(relpos, (3, 1))
    steps_21 = np.tile(1 - relpos, (3, 1))  # reverse weights for surface 2

    # Construct a 4-dimensional matrix
    cs_rep = np.zeros((3, steps, ncs, 2))
    for i in range(steps):
        cs_rep[:, i, :, 0] = cs1
        cs_rep[:, i, :, 1] = cs2

    # Weighting matrix
    steps_rep = np.zeros((3, steps, ncs, 2))
    steps_rep[:, :, :, 0] = np.tile(steps_12[:, :, np.newaxis], (1, 1, ncs))
    steps_rep[:, :, :, 1] = np.tile(steps_21[:, :, np.newaxis], (1, 1, ncs))

    # Multiply coordinates with weights and sum
    CS = np.sum(steps_rep * cs_rep, axis=3)

    return CS

import numpy as np

def surfing_uniqueidxsperrow(x):
    """
    Returns unique indices per row.

    Parameters:
    x : ndarray
        NxPxQ array, where N is the number of rows, P is the number of steps, and Q is the number of nodes.

    Returns:
    y : ndarray
        NxMxQ array, M<=P, where each row contains the unique
        values in the corresponding row from x.
        Only positive values in x are returned, and
        each row may contain zeros in case duplicate values
        were found in that row.
    """
    if x.ndim != 3:
        raise ValueError("Input array must be 3-dimensional")

    nrows, nsteps, nnodes = x.shape
    y = np.zeros((nrows, nsteps, nnodes))

    for k in range(nrows):
        for j in range(nnodes):
            # Only consider positive elements, and take the unique values
            unqxk = np.unique(x[k, :, j][x[k, :, j] > 0])
            # Store them in y
            y[k, :len(unqxk), j] = unqxk

    # Determine the necessary number of columns
    max_nonzero_cols = 0
    for k in range(nrows):
        for j in range(nnodes):
            nonzero_cols = np.count_nonzero(y[k, :, j])
            if nonzero_cols > max_nonzero_cols:
                max_nonzero_cols = nonzero_cols

    y = y[:, :max_nonzero_cols, :]
    return y


def readSurf(white_files,pial_files):
# def readSurf(white_files,pial_files, names=['L','R']):
    # white: list of file names of white matter surface, e.g., 'fs_LR.32k.L.white.surf.gii'
    # pial: list of file names of pial surface, e.g., 'fs_LR.32k.L.pial.surf.gii'
    # Read a gifti file
    white = [nb.gifti.GiftiImage.from_filename(f) for f in white_files]
    pial = [nb.gifti.GiftiImage.from_filename(f) for f in pial_files]
    S = []
    for i in range(len(white)):
        S.append({'topo':white[i].darrays[1], 'white': white[i].darrays[0], 'pial': pial[i].darrays[0]})

    # for i, name in enumerate(names):
    #     S[name, 'topo'] = white[i].darrays[1]
    #     S[name, 'white'] = white[i].darrays[0]
    #     S[name, 'pial'] = pial[i].darrays[0]
    return S


def surfing_circle_roi(self, ca, n2f, node_idx, radius, distance_metric):
    dists = np.linalg.norm(ca.T - ca[:, node_idx].reshape(-1, 1), axis=0)
    node_idxs = np.where(dists <= radius)[0]
    node_dist = dists[node_idxs]
    return node_idxs, node_dist

def surfing_coords2linvoxelidxs(self, all_coords):
    ijk = all_coords.T
    lin_vox_idxs = np.ravel_multi_index(ijk, self.Mask.shape, order='F')
    return lin_vox_idxs

def surfing_uniqueidxsperrow(self, all_lin_vox_idxs):
    unq_lin_vox_idxs = np.empty_like(all_lin_vox_idxs)
    for i, row in enumerate(all_lin_vox_idxs):
        unq_lin_vox_idxs[i] = np.unique(row)
    return unq_lin_vox_idxs

def surfing_eucldist(self, center):
    dists = np.linalg.norm(self.ca.T - center, axis=1)
    return dists

def surfing_inds2subs(self, shape, idxs):
    subs = np.unravel_index(idxs, shape, order='F')
