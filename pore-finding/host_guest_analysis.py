import copy
import freud
import numpy as np
import rowan


def get_patch_nlist(box, patch_locs, lambdasigma, host_pos, host_ors):
    # Store positions and orientations of patch locations in correct array format
    patch_ors = np.repeat(host_ors, len(patch_locs), axis=0)
    patch_pos = np.repeat(host_pos, len(patch_locs), axis=0)

    # Collect, rotate, and adjust patch locations to represent proper placement
    frame_patch_locations = np.tile(patch_locs, [len(host_pos), 1])
    frame_patch_locations = rowan.rotate(patch_ors, frame_patch_locations)
    frame_patch_locations += patch_pos

    # AABB Query patch locations to find pairs of neighbores
    nq = freud.AABBQuery(box, frame_patch_locations)
    query_result = nq.query(frame_patch_locations,
                            dict(r_max=lambdasigma, exclude_ii=True))
    nlist = query_result.toNeighborList()
    num_patches = len(patch_locs)
    host_nlist = np.array([(i // num_patches, j // num_patches)
                           for i, j in nlist])
    return host_nlist


def create_adjacency_matrix(nlist, num_particles):
    """Create an adjacency matrix describing bonds between particles

    Args
    ----
    nlist : np.array, shape=(n,2), dtype=int
        List of which particles are bonded to which
    num_particles : int
        The number of particles that can bond to each other in the system

    Returns
    -------
    adjacency_matrix : np.ndarray, shape=(num_particles, num_particles)
        The adjacency matrix

    """
    adj_matrix = np.zeros((num_particles, num_particles), dtype=np.uint8)
    for i, j in nlist:
        adj_matrix[i][j] = 1
        adj_matrix[j][i] = 1
    return adj_matrix


def find_hexamer(adj_matrix, passed_rows, row, hexamer, hexamers):
    """Recursively find a ring of bonded particles

    Note
    ----
    This function returns double the number of actual rings (or pores/hexamers),
    and I can't figure out how to fix it. But that's OK (for now, at least).

    """
    # base case: a loop is discovered -> add hexamer to hexamers
    # TODO: don't add hexamer to hexamers if it's already in there
    if len(hexamer) == 5 and adj_matrix[row][hexamer[0]] == 1:
        hexamer.append(row)
        hexamers.append(hexamer)
        return
    # base case: hexamer is size 6, but no loop -> return without doing anything
    elif len(hexamer) >= 5:
        return
    # neither base case is satisfied -> add current row to hexamer and
    # move to recursive case
    else:
        hexamer.append(row)

    # FIX: does not account for certain cases
    # last_row = row
    for idx in range(passed_rows + 1, len(adj_matrix[row])):
        x = adj_matrix[row][idx]
        if x == 1 and not (idx in hexamer):
            copy_of_hexamer = copy.deepcopy(hexamer)
            find_hexamer(
                adj_matrix,
                passed_rows,
                idx,
                copy_of_hexamer,
                hexamers,
            )


def find_hexamers(adjacency_matrix):
    """Find hexamers from the adjacency matrix

    This function calls the recursive function find_hexamer.

    Args
    ----
    adjacency_matrix : np.ndarray, shape=(n, n)
        The adjacency matrix

    Returns
    -------
    hexamers : np.ndarray, shape=(n, 6), dtype=int
        The list of list of particles indices that make up hexamers

    """
    passed_rows = 0
    hexamers = []
    for idx, row in enumerate(adjacency_matrix):
        find_hexamer(adjacency_matrix, passed_rows, idx, [], hexamers)
        passed_rows += 1
    return hexamers


def get_pore_centers(hexamers, host_pos, box):
    """Locate the centers of the pores.

    The values in `hexamers` corresponds to indices in the `host_positions`
    array, not in the global array of particle positions.
    Each row in `hexamers` gives the indices of particles that form a pore.
    
    While not necessary, it may be helpful to remove any repeated entries in
    `hexamers` before passing it to this function.

    Note that this function calculates the center of the pore as the halfway
    point between the 0th and 3rd particle in each hexamer. It is therefore
    important that the indices in each row of `hexamers` are ordered in the same
    order that the particles are bonded to each others.

    Args
    ----
    hexamers : np.ndarray, shape=(n_pores, 6), dtype=int
        Particle indices that form hexamers
    host_positions : np.ndarray, shape=(n, 3), dtype=float
        Positions of host particles
    box : freud.Box
        The simulation box

    Returns
    -------
    pore_centers : np.ndarray, shape=(n_pores, 3), dtype=float
        The cooradinaes of the centers of the pores
    max_circumcircle_radius : float
        The largest circumcircle radius of all pores

    """
    pore_centers = []
    max_r_circ = -np.inf  # maximum circumcircle radius of all pores
    for hexamer in hexamers:
        com = 0.5 * (box.wrap(host_pos[hexamer[3]] -
                              host_pos[hexamer[0]])) + host_pos[hexamer[0]]
        max_r_circ = max(
            max_r_circ,
            max(np.linalg.norm(box.wrap(host_pos[hexamer] - com), axis=1)))
        pore_centers.append(com)
    return np.array(pore_centers), max_r_circ


def find_guests_in_pores(guest_pos, pore_centers, search_radius, box):
    """Find guests that are inside pores.

    Args
    ----
    guest_pos : np.ndarray, shape=(num_guests, 3), dtype=float
        Guest positions
    pore_centers : np.ndarray, shape=(num_pores, 3), dtype=float
        Locations of pore centers
    search_radius : float
        Consider guests inside pore if it is less than search_radus from the
        center of the pore
    box : freud.Box
        The simulation box

    Returns
    -------
    captured_guest_idxs : np.array, shape=(num_captured_guests,), dtype=int
        The guest_pos indices of guests that are inside of a pore

    """
    aabbq = freud.AABBQuery(box, pore_centers)
    qargs = dict(mode="ball", exclude_ii=False, r_max=search_radius)
    guest_hexamer_nlist = aabbq.query(guest_pos, qargs).toNeighborList()
    captured_guest_idxs = guest_hexamer_nlist[:, 0]
    return captured_guest_idxs


def find_hexamer_hosts(box, patch_locations, patch_range, host_positions,
                       host_orientations):
    """Find host particles that are a part of a pore hexamer

    Args
    ----
    box : freud.Box
        The box
    patch_locations : np.ndarray, shape=(n,3)
        The locations of the patches in the particle frame of reference
    host_positions : np.ndarray, shape=(n,3)
        The positions of the host particles
    host_orientations : np.ndarray, shape=(n,4)
        The orientations of the host particles

    Returns
    -------
    hosts : np.ndarray, len=(n,), dtype=int
        The indices of the host particles that are a part of a pore motif
    """
    host_host_nlist = get_patch_nlist(
        box,
        patch_locations,
        patch_range,
        host_positions,
        host_orientations,
    )
    adj_matrix = create_adjacency_matrix(host_host_nlist, len(host_positions))


if __name__ == '__main__':
    import signac
    import gsd.hoomd
    #import cProfile
    import subprocess
    import pprofile
    import time

    project = signac.get_project('../')
    job = project.open_job(id='cc06')
    filename = job.fn('traj.gsd')
    with gsd.hoomd.open(filename, 'rb') as gsd_file:
        for frame in gsd_file[20:21]:
            # get stuff
            pos = frame.particles.position
            ors = frame.particles.orientation
            Lx, Ly, Lz, xy, xz, yz = frame.configuration.box
            box = freud.Box(Lx=Lx, Ly=Ly, xy=xy, is2D=True)
            typeid_A = frame.particles.types.index('A')
            typeid_B = frame.particles.types.index('B')
            host_idxs = np.argwhere(
                frame.particles.typeid == typeid_A).flatten()
            guest_idxs = np.argwhere(
                frame.particles.typeid == typeid_B).flatten()
            host_positions = pos[host_idxs]
            host_orientations = ors[host_idxs]
            guest_positions = pos[guest_idxs]
            patch_range = job.sp.lambdasigma
            patch_locations = job.doc.patch_locations[0]

            # create host-host neighborlist and adjacency matrix
            host_host_nlist = get_patch_nlist(
                box,
                patch_locations,
                patch_range,
                host_positions,
                host_orientations,
            )
            adj_matrix = create_adjacency_matrix(
                host_host_nlist,
                len(host_positions),
            )

            # find hexamers from adjacency matrix
            start_time = time.time()
            hexamers = find_hexamers(adj_matrix)
            end_time = time.time()
            print(f'find_hexamers() took {end_time - start_time} seconds')
            hosts_in_hexamers = np.unique(np.array(hexamers, dtype=np.int64).flatten())
            global_idxs_hosts_in_hexamers = host_idxs[hosts_in_hexamers]

            # exit if no pores found
            if len(hexamers) == 0:
                break

            # find pore centers
            _, indices = np.unique(
                np.sort(np.array(hexamers)),
                axis=0,
                return_index=True,
            )
            uniqe_hexamers = np.array(hexamers)[indices]
            pore_centers, search_radius = get_pore_centers(
                uniqe_hexamers,
                host_positions,
                box,
            )

            # find captured guests
            captured_guest_idxs = find_guests_in_pores(
                guest_positions,
                pore_centers,
                search_radius,
                box,
            )
            global_idxs_guests_in_pores = guest_idxs[captured_guest_idxs]

            # select particles for visualization
            hexamer_host_sel = ''.join([
                f'ParticleIndex == {x} || '
                for x in global_idxs_hosts_in_hexamers
            ])
            captured_guest_sel = ''.join([
                f'ParticleIndex == {x} || '
                for x in global_idxs_guests_in_pores
            ])

            subprocess.run(
                "pbcopy",
                universal_newlines=True,
                input=hexamer_host_sel,
            )
