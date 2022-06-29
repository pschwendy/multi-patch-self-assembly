import gsd.hoomd
import freud
import numpy as np


# reading gsd files is very straightforward, you only need the gsd package see
# https://gsd.readthedocs.io/en/v2.5.2/ for info (it's pip- and
# conda-installable)
gsd_filename = 'workspace/8e7d9da1df4b075f4558acff994e819b/traj.gsd'

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
Here is a function I've used in the past to find patchy particles that are bonded.
It works by creating a neighborlist of patches that are within the interaction
range, which makes them bonded.
"""
def get_patch_nlist(frame, patch_locs, lambdasigma):
    pos = frame.positions
    ors = frame.orientations
    patch_ors = np.repeat(frame.orientations, len(patch_locs), axis=0)
    patch_pos = np.repeat(frame.positions, len(patch_locs), axis=0)
    frame_patch_locations = np.tile(patch_locs, [frame.N, 1])
    frame_patch_locations = rowan.rotate(patch_ors, frame_patch_locations)
    frame_patch_locations += patch_pos
    nq = freud.AABBQuery(get_box(frame), frame_patch_locations)
    query_result = nq.query(frame_patch_locations, dict(r_max=lambdasigma, exclude_ii=True))
    return query_result.toNeighborList()
