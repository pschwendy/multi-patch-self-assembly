import gsd.hoomd
import freud
import numpy as np
import rowan

# TODO: create a job looper to access gsd files of each job
# TODO: locate pores through patch bonds
# TODO: determine whether 


# reading gsd files is very straightforward, you only need the gsd package see
# https://gsd.readthedocs.io/en/v2.5.2/ for info (it's pip- and
# conda-installable)
# gsd_filename = 'workspace/8e7d9da1df4b075f4558acff994e819b/traj.gsd'

"""
# first open the file with gsd.hoomd:
with gsd.hoomd.open(gsd_filename, 'rb') as traj:
    # the simply iterate over the open file
    for frame in traj:
        # get box information from gsd file and construct a freud box with it,
        # so that we can easily get things like the volume (area in 2D)
        Lx, Ly, Lz, xy, xz, yz = frame.configuration.box
        box = freud.Box(Lx=Lx, Ly=Ly, xy=xy, is2D=True)

        # particle position and orientation arrays are aptly named...
        pos = frame.particles.position
        ors = frame.particles.orientation

        # many freud analysis routines take in the system info as a (box,
        # points) tuple, so I generally just construct that right away
        system = (box, pos)

        # frame.particles.types is the types of particles in the system
        # for example, for our 2-component systems, it would return ['A', 'B']
        ptypes = frame.particles.types

        # frame.particles.typeid gives the integer particle type, and the string
        # type name is the item in frame.particles.types with the index of the
        # typeid.  To get the positions of all of the type 'A' particles in the
        # system, we just take a slice of the positions array where the typeid
        # corresponds to the index of frame.particles.types that returns 'A':
        typeid_A = frame.particles.types.index('A')
        typeid_B = frame.particles.types.index('B')
        pos_A = pos[frame.particles.typeid == typeid_A]
        ors_A = ors[frame.particles.typeid == typeid_A]
        pos_B = pos[frame.particles.typeid == typeid_B]
        ors_B = ors[frame.particles.typeid == typeid_B]

        # now we have separate arrays that contain the positions of just
        # particles of type 'A' and type 'B' separately
        # here's an example of how to find neighbors in freud: for each 'B'
        # particle, we will find the 6 nearest 'A' particles
        # first, we create an axes-aligned bounding box-based neighborlist
        # object using the 'A' particles to build the tree
        aabbq = freud.AABBQuery(box, pos_A)

        # create a dict of query arguments. we want to find n-nearest neighbors,
        # so we set mode='nearest' and num_neighbors=6. we also do not
        # exclude_ii since our reference points ('A') and query points ('B') are
        # disjoint sets
        qargs = dict(mode='nearest', num_neighbors=6, exclude_ii=False)

        # now we can query our AABB tree with the 'B' points. per the freud
        # docs, "freud promises that every query_point will end up with
        # num_neighbors points,"
        # (https://freud.readthedocs.io/en/stable/topics/querying.html) which is
        # a helpful way to remember which points to use as the query points
        # aabbq.query() returns a NeighborQueryResult, which we immediately
        # convert to a proper NeighborList object
        nlist = aabbq.query(pos_B, qargs).toNeighborList()
        
        # now we can do whatever we need to with the neighborlist.
        # see
        # https://freud.readthedocs.io/en/stable/modules/locality.html#freud.locality.NeighborList
        # for methods that it has. 
        # to get vectors from the 'B' particles to the 'A' ones, we just use the
        # indices in the neighborlist. e.g.:
        for i, j in nlist:
            r_ij = pos_A[j] - pos_B[i]
            print(r_ij)
            break
"""

"""
Here is a function I've used in the past to find patchy particles that are bonded.
It works by creating a neighborlist of patches that are within the interaction
range, which makes them bonded.
"""
# TODO: pass list of host orientations and positions
def get_patch_nlist(frame, box, patch_locs, lambdasigma, host_pos, host_ors):   
    # Store positions and orientations of patch locations in correct array format
    patch_ors = np.repeat(host_ors, len(patch_locs), axis=0)
    patch_pos = np.repeat(host_pos, len(patch_locs), axis=0)
    
    # Collect, rotate, and adjust patch locations to represent proper placement
    frame_patch_locations = np.tile(patch_locs, [len(host_pos), 1])
    frame_patch_locations = rowan.rotate(patch_ors, frame_patch_locations)
    frame_patch_locations += patch_pos
    
    # AABB Query patch locations to find pairs of neighbores
    nq = freud.AABBQuery(box, frame_patch_locations)
    query_result = nq.query(frame_patch_locations, dict(r_max=lambdasigma, exclude_ii=True))

    nlist = query_result.toNeighborList()
    host_nlist = []
    for i, j in nlist:
        idx_i = i // 3
        idx_j = j // 3
        host_nlist.append((idx_i, idx_j))
    print(f"num hosts: {len(host_pos)}")
    print(f"len host_nlist: {len(host_nlist)}")
    return host_nlist


"""
nlist = []
for i, j in patch_nlist:
    host_idx_i = i // len(patch_locs)
    host_idx_j = j // len(patch_locs)
    nlist.append[(host_idx_i, host_idx_j)]


"""

"""
Algorithm for finding hexamers

1. Override query list as list of pairs of particles.

2. Form an n x n adjacency matrix, where n is the number of particles. 
Populate the matrix with 1s and 0s, where 1 marks a pair and 0 signifies no relationship
For example, if particles A and B form a pair, set matrix[A][B] = 1 and matrix[B][A] = 1

Subsequently, form a hash map of pair to int, set each pair and it's conjugate to point to 0
For example, map[(a, b)] = 0 and map[(b, a)] = 0

3. 
"""
import copy
# idx = 0
# 0, 1, 5, 2, 3, 6
# 0, 6, 3, 2, 5, 1
# [0][1] = 1, [0][6] = 1
# try set of pairs (?)
def find_hexamer(adj_matrix, passed_rows, row, hexamer, hexamers):
    # base case: a loop is discovered -> add hexamer to hexamers
    if len(hexamer) == 5 and adj_matrix[row][hexamer[0]] == 1:
        hexamer.append(row)
        hexamers.append(hexamer)
        return
    # base case: hexamer is size 6, but no loop -> return without doing anything
    elif len(hexamer) >= 5:
        return
    # neither base case is satisfied -> add current row to hexamer and move to recursive case
    else:
        hexamer.append(row)

    # FIX: does not account for certain cases
    # last_row = row
    for idx in range(passed_rows + 1, len(adj_matrix[row])):
        x = adj_matrix[row][idx]
        # if len(hexamer) == 5 and hexamer[4] == 1657 and idx == 1796:
        if x == 1 and not (idx in hexamer):
            copy_of_hexamer = copy.deepcopy(hexamer)
            find_hexamer(adj_matrix, passed_rows, idx, copy_of_hexamer, hexamers)
            

def calc_loaded_pores(frame, nlist, num_particles):
    adj_matrix = np.zeros((num_particles, num_particles))
    passed_rows = 0

    for i, j in nlist:
        adj_matrix[i][j] = 1
        adj_matrix[j][i] = 1

    hexamers = []

    # [822, 1670, 1876, 2245, 1657, 1796]
    for idx, row in enumerate(adj_matrix):
        find_hexamer(adj_matrix, passed_rows, idx, [], hexamers)
        passed_rows += 1

    # print(hexamers)
    
    Lx, Ly, Lz, xy, xz, yz = frame.configuration.box
    box = freud.Box(Lx=Lx, Ly=Ly, xy=xy, is2D=True)
    
    typeid_A = frame.particles.types.index('A')
    host_pos = pos[frame.particles.typeid == typeid_A]

    typeid_B = frame.particles.types.index('B')
    guest_pos = pos[frame.particles.typeid == typeid_B]

    hexamer_center_locs = []
    r_circ = -np.inf
    for hexamer in hexamers:
        # com = np.mean(host_pos[hexamer], axis=0)
        com = (0.5 * (box.wrap(host_pos[hexamer[3]] - host_pos[hexamer[0]]))) + host_pos[hexamer[0]]
        # print(max(np.linalg.norm(host_pos[hexamer], axis=1)))
        r_circ = max(r_circ, max(np.linalg.norm(box.wrap(host_pos[hexamer] - com), axis=1)))
        if r_circ > 50:
            breakpoint()
        hexamer_center_locs.append(com)
    
    hexamers_center_locs = np.array(hexamer_center_locs)
    aabbq = freud.AABBQuery(box, hexamers_center_locs)
    # breakpoint()
    print(f"r_circ: {r_circ}")
    qargs = dict(mode="ball", exclude_ii=False, r_max=r_circ)
    guest_hexamer_nlist = aabbq.query(guest_pos, qargs).toNeighborList()
    
    print(f'num guests = {len(guest_pos)}')
    n_captured_guest = len(guest_hexamer_nlist)
    print(f'num captured guests = {n_captured_guest}')
    n_hexamers = len(hexamers)
    print(f'num hexamers = {n_hexamers}')

    fractional_loading = n_captured_guest / n_hexamers
    return fractional_loading
    # numpy mean of position 
    

import signac
project = signac.get_project()
job = project.open_job(id='cc06')
filename = job.fn('traj.gsd')
with gsd.hoomd.open(filename, 'rb') as gsd_file:
    for frame in gsd_file[-1:]:
        pos = frame.particles.position
        ors = frame.particles.orientation

        Lx, Ly, Lz, xy, xz, yz = frame.configuration.box
        box = freud.Box(Lx=Lx, Ly=Ly, xy=xy, is2D=True)
        
        typeid_A = frame.particles.types.index('A')
        host_pos = pos[frame.particles.typeid == typeid_A]
        host_ors = ors[frame.particles.typeid == typeid_A]
        print(f'patch range = {job.sp.lambdasigma}')
        nlist = get_patch_nlist(frame, box, job.doc.patch_locations[0], job.sp.lambdasigma, host_pos, host_ors)
        fractional_loading = calc_loaded_pores(frame, nlist, len(host_pos))
        print(f'fractional loading = {fractional_loading}')
