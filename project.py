import flow
from flow import FlowProject, directives, cmd
from flow import environments
import itertools
import math
import numpy as np
import os
from random import shuffle
import rowan
import string

import patch_c_code


class Project(FlowProject):

    def __init__(self):
        flow.FlowProject.__init__(self)
        self.required_sp_params = (
            'phi',
            'initial_config',
            'replica',
            'n_repeats',
            # 'betaP',
            'use_floppy_box',
            'patch_offset',  # 0 = middle, ±1 = on vertex
            'n_host_edges',
            'n_guest_edges',
            'sigma',
            'lambdasigma',
            'kT',
            'guest_info',  # [(relative size, mol_fraction),...]
        )
        self.required_defaults = (
            ('gsd_frequency', int(1e5)),
            ('thermo_frequency', int(1e4)),
            ('n_tune_blocks', 20),
            ('n_tune_steps', 10),
            ('n_run_steps', int(3e5)),
            ('compress_scale', 0.9999),
            ('stop_after', int(50e6)),
        )

    @staticmethod
    def generate_patch_location_c_code(job):
        ret_str = ''
        for particle in job.doc.patch_locations:
            ret_str += "{"
            for pl in particle:
                ret_str_init = 'vec3<float>({}),\n'.format(', '.join(
                    map(str, pl)))
                if pl is particle[-1]:
                    ret_str_init = ret_str_init[:len(ret_str_init) - 2] + "\n"
                ret_str += ret_str_init
            if particle is not job.doc.patch_locations[-1]:
                ret_str += "},\n"
            else:
                ret_str += "}"
        return ret_str


def done_running(job):
    try:
        end_time = job.doc.stop_after
        cr = job.doc.get('continue_running', True)
        ts_criterion = job.doc.get('timestep', 0) > end_time
        stopping_criteria = (not cr, ts_criterion)
        return any(stopping_criteria)
    except:
        return False


@Project.label
def timestep_label(job):
    if done_running(job):
        return 'done running'
    ts = job.doc.get('timestep', None)
    if ts is None:
        return False
    ts = str(ts)
    return '{}.{}{}e{}'.format(ts[0], ts[1], ts[2], len(ts) - 1)


@Project.operation
@Project.post.isfile('init.gsd')
def initialize(job):
    import hoomd
    import hoomd.hpmc
    import scipy.spatial
    """Sets up the system and sets default values for writers and stuf

    """
    # sanity check: guest mol fractions must sum to less than 1
    print(job.sp.guest_info)
    total_xg = sum([x[1] for x in job.sp.guest_info])
    if total_xg >= 1.0:
        raise ValueError('Too many guests in {}'.format(job._id[:6]))

    # initialize the hoomd context
    msg_fn = job.fn('init-hoomd.log')
    hoomd.context.initialize('--mode=cpu --msg-file={}'.format(msg_fn))

    # figure out shape vertices and patch locations
    n_e = job.sp.n_host_edges
    xs = np.array([np.cos(n * 2 * np.pi / n_e) for n in range(n_e)])
    ys = np.array([np.sin(n * 2 * np.pi / n_e) for n in range(n_e)])
    zs = np.zeros_like(ys)
    host_vertices = np.vstack((xs, ys, zs)).T

    # create the guest vertices...they'll be scaled appropriately later
    n_guest_edges = job.sp.n_guest_edges
    xs = np.array(
        [np.cos(n * 2 * np.pi / n_guest_edges) for n in range(n_guest_edges)])
    ys = np.array(
        [np.sin(n * 2 * np.pi / n_guest_edges) for n in range(n_guest_edges)])
    zs = np.zeros_like(ys)
    guest_vertices = np.vstack((xs, ys, zs)).T

    # build the system
    n_repeats = job.sp.n_repeats
    if job.sp.initial_config == 'dilute':
        uc = hoomd.lattice.sq(2.0, type_name='A')
        system = hoomd.init.create_lattice(uc, n_repeats)
    else:
        raise NotImplementedError('Initialization not implemented.')

    # scale edges to have unit length edges
    edge_length = np.linalg.norm(host_vertices[1] - host_vertices[0])
    host_area = scipy.spatial.ConvexHull(host_vertices[:, :2]).volume
    host_vertices = host_vertices - np.mean(host_vertices, axis=0)
    vertex_vertex_vectors = np.roll(host_vertices, -1, axis=0) - host_vertices
    half_edge_locations = host_vertices + 0.5 * vertex_vertex_vectors
    patch_locations = []
    for patch_offset, letter in zip(job.sp.patch_offset,
                                    string.ascii_uppercase):
        f = patch_offset
        pl_np = half_edge_locations + f * (host_vertices - half_edge_locations)
        patch_locations.append(pl_np.tolist())
        system.particles.types.add(letter)

    # change types of hosts
    snapshot = system.take_snapshot(all=True)
    indices = np.arange(len(system.particles), dtype=np.int)
    shuffle(indices)
    start_idx = 0
    n_guest_particles = int(
        sum([x[1] for x in job.sp.guest_info]) * len(system.particles))
    n_host_particles = int(len(system.particles) - n_guest_particles)
    for patch_offset, ptype in zip(job.sp.patch_offset,
                                   system.particles.types):
        n_this_type = int(n_host_particles / len(job.sp.patch_offset))
        type_idx = snapshot.particles.types.index(ptype)
        end_idx = start_idx + n_this_type
        for p_idx in indices[start_idx:end_idx]:
            snapshot.particles.typeid[p_idx] = type_idx
        start_idx = end_idx
    system.restore_snapshot(snapshot)

    # add the different guests
    guest_particle_area = scipy.spatial.ConvexHull(
        guest_vertices[:, :2]).volume
    guests, guest_areas = {}, {}
    ptypes = [x for x in string.ascii_uppercase[len(job.sp.patch_offset):]]
    for rs, xg in job.sp.guest_info:
        particle_area = host_area * rs
        ptype = ptypes.pop(0)
        mult_factor = np.sqrt(particle_area / guest_particle_area)
        print(host_area, rs, particle_area, guest_particle_area, mult_factor)
        new_guest_vertices = guest_vertices * mult_factor
        print(scipy.spatial.ConvexHull(new_guest_vertices[:, :2]).volume)
        guests[ptype] = new_guest_vertices
        guest_areas[ptype] = particle_area
        system.particles.types.add(ptype)

    # change some of the particle types
    snapshot = system.take_snapshot(all=True)
    indices = np.arange(len(system.particles), dtype=np.int)
    shuffle(indices)
    start_idx = 0
    for (rs, xg), ptype in zip(job.sp.guest_info, guests.keys()):
        n_this_type = int(xg * len(system.particles))
        type_idx = snapshot.particles.types.index(ptype)
        end_idx = start_idx + n_this_type
        for p_idx in indices[start_idx:end_idx]:
            snapshot.particles.typeid[p_idx] = type_idx
        start_idx = end_idx
    system.restore_snapshot(snapshot)

    # restart writer; period=None since we'll just call write_restart() at end
    restart_writer = hoomd.dump.gsd(filename=job.fn('restart.gsd'),
                                    group=hoomd.group.all(),
                                    truncate=True,
                                    period=None,
                                    phase=0)

    # set up the integrator with the shape info
    seed = job.sp.replica
    mc = hoomd.hpmc.integrate.convex_polygon(seed=seed, d=0, a=0)
    # mc.shape_param.set('A', vertices=host_vertices[:, :2])
    for patch_offset, letter in zip(job.sp.patch_offset,
                                    string.ascii_uppercase):
        mc.shape_param.set(letter, vertices=host_vertices[:, :2])
    for ptype, vertices in guests.items():
        mc.shape_param.set(ptype, vertices=vertices[:, :2])
    restart_writer.dump_shape(mc)
    restart_writer.dump_state(mc)
    hoomd.run(1)
    mc.set_params(d=0.1, a=0.5)

    # save everything into the job doc that we need to
    if hoomd.comm.get_rank() == 0:
        job.doc.mc_d = {x: 0.1 for x in system.particles.types}
        job.doc.mc_a = {x: 0.1 for x in system.particles.types}
        job.doc.host_vertices = host_vertices
        job.doc.guest_vertices = guests
        job.doc.guest_areas = guest_areas
        job.doc.patch_locations = patch_locations
        job.doc.host_area = host_area
        job.doc.guest_areas = guest_areas
        for k, v in job._project.required_defaults:
            job.doc.setdefault(k, v)
    hoomd.comm.barrier()
    restart_writer.write_restart()
    if hoomd.comm.get_rank() == 0:
        os.system('cp {} {}'.format(job.fn('restart.gsd'), job.fn('init.gsd')))
    hoomd.comm.barrier()
    return


def calculate_temp(job, timestep):
    if len(job.sp.kT_ramp) == 3:
        kT_0, kT_f, length = job.sp.kT_ramp
        calculated_kT = (kT_f - kT_0) / length * timestep + kT_0
        return max(calculated_kT, kT_f)
    elif len(job.sp.kT_ramp) == 5:
        kT_const, dT_const, kT_0, kT_f, length = job.sp.kT_ramp
        if timestep < dT_const:
            return kT_const
        else:
            calculated_kT = (kT_f - kT_0) / length * (timestep - dT_const)
            calculated_kT += kT_0
            return max(calculated_kT, kT_f)
    else:
        raise ValueError('Invalid kT_ramp in job statepoint.')


@Project.operation
@Project.pre.after(initialize)
@Project.post(done_running)
@directives(nranks=16)
@directives(
    executable='singularity exec software.simg python3',
    walltime=2,
)
def sample(job):
    import hoomd
    import hoomd.hpmc
    import hoomd.jit
    # get all parameters first here
    seed = job.sp.replica
    use_floppy_box = job.sp.use_floppy_box
    gsd_frequency = job.doc.gsd_frequency
    thermo_frequency = job.doc.thermo_frequency
    n_tune_blocks = job.doc.n_tune_blocks
    n_tune_steps = job.doc.n_tune_steps
    n_run_steps = job.doc.n_run_steps
    mc_d, mc_a = job.doc.mc_d, job.doc.mc_a
    host_vertices = np.array(job.doc.host_vertices)[:, :2]
    scale = job.doc.compress_scale
    last_tuning_kT = job.doc.get('last_tuning_kT', np.inf)
    final_step = job.doc.stop_after

    # handle hoomd message files: save old output to a file
    msg_fn = job.fn('hoomd-log.txt')
    hoomd.context.initialize('--mode=cpu')
    if os.path.isfile(msg_fn):
        if hoomd.comm.get_rank() == 0:
            with open(job.fn('previous-hoomd-logs.txt'), 'a') as outfile:
                with open(msg_fn, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
    hoomd.comm.barrier()
    hoomd.option.set_msg_file(msg_fn)

    # initialize system from restart file
    initial_gsd_fn = job.fn('init.gsd')
    restart_fn = job.fn('restart.gsd')
    system = hoomd.init.read_gsd(initial_gsd_fn, restart_fn)

    gsd_writer = hoomd.dump.gsd(job.fn('traj.gsd'),
                                gsd_frequency,
                                hoomd.group.all(),
                                overwrite=False,
                                truncate=False)
    restart_frequency = gsd_frequency
    restart_writer = hoomd.dump.gsd(filename=restart_fn,
                                    group=hoomd.group.all(),
                                    truncate=True,
                                    period=restart_frequency,
                                    phase=0)
    mc = hoomd.hpmc.integrate.convex_polygon(seed=seed, restore_state=True)
    for patch_offset, letter in zip(job.sp.patch_offset,
                                    string.ascii_uppercase):
        mc.shape_param.set(letter, vertices=host_vertices[:, :2])

    # add guest particles
    for ptype, vertices in job.doc.get('guest_vertices', {}).items():
        mc.shape_param.set(ptype, vertices=np.array(vertices)[:, :2])
    mc_tuners = {
        t: hoomd.hpmc.util.tune(mc,
                                tunables=['d', 'a'],
                                target=0.33,
                                gamma=0.5,
                                type=t)
        for t in system.particles.types
    }
    restart_writer.dump_state(mc)
    gsd_writer.dump_state(mc)
    restart_writer.dump_shape(mc)
    gsd_writer.dump_shape(mc)
    logger_info = {
        'thermo.txt': ['volume', 'lx', 'ly', 'lz', 'xy'],
        'hpmc-stats.txt': [
            'hpmc_sweep', 'hpmc_translate_acceptance',
            'hpmc_rotate_acceptance', 'hpmc_d', 'hpmc_a'
        ],
        'hpmc-patch-stats.txt': ['hpmc_patch_energy', 'hpmc_patch_rcut']
    }
    loggers = {}
    for fn, quantities in logger_info.items():
        loggers[fn[:-4]] = hoomd.analyze.log(job.fn(fn),
                                             quantities,
                                             thermo_frequency,
                                             header_prefix='# ',
                                             overwrite=False)

    # compress if we're not at the target packing fraction
    N_types = {ptype: 0 for ptype in system.particles.types}
    for p in system.particles:
        N_types[p.type] += 1
    A_particles = 0
    for ptype, count in N_types.items():
        area = job.doc.guest_areas.get(ptype, job.doc.host_area)
        A_particles += area * count
    A_target = A_particles / job.sp.phi
    L_current = np.array([system.box.Lx, system.box.Ly])
    L_target = L_current * np.sqrt(A_target / system.box.get_volume())
    phi = A_particles / system.box.get_volume()
    n_expand_steps = 0
    need_to_compress = not job.doc.get('compressed', False)
    if need_to_compress:
        hoomd.run(1)
    while (not math.isclose(phi, job.sp.phi)) and need_to_compress:
        L = np.maximum(L_current * scale, L_target)
        hoomd.update.box_resize(Lx=L[0], Ly=L[1], period=None)
        L_current = np.array([system.box.Lx, system.box.Ly])
        n_overlaps = mc.count_overlaps()
        phi = A_particles / system.box.get_volume()
        hoomd.context.msg.notice(
            1, 'phi = {:.3f}; {} overlaps\n'.format(phi, n_overlaps))
        n_overlap_remove_steps = 0
        while n_overlaps > 0:
            hoomd.run(100, quiet=True)
            n_overlaps = mc.count_overlaps()
            n_overlap_remove_steps += 1
            if n_overlap_remove_steps > 10:
                L_exp = np.array([system.box.Lx, system.box.Ly, system.box.Lz
                                  ]) * 1.01
                hoomd.update.box_resize(Lx=L_exp[0],
                                        Ly=L_exp[1],
                                        Lz=L_exp[2] / 1.01,
                                        period=None)
                n_expand_steps += 1
            L_current = np.array([system.box.Lx, system.box.Ly])
        if n_expand_steps > 10:
            if hoomd.comm.get_rank() == 0:
                job.doc['trouble_compressing'] = True
                job.doc['compressed'] = False
            hoomd.comm.barrier()
            return
        hoomd.context.msg.notice(1, '\n')
    if hoomd.comm.get_rank() == 0:
        job.doc.compressed = True
        job.doc.setdefault('compressed_timestep', hoomd.get_step())
    hoomd.comm.barrier()

    # patches
    snapshot = system.take_snapshot()
    host_type_indexes = [
        snapshot.particles.types.index(letter) for patch_offset, letter in zip(
            job.sp.patch_offset, string.ascii_uppercase)
    ]
    host_type_idx = "{"
    for x in host_type_indexes:
        host_type_idx += f"{x}"
        if x is not host_type_indexes[-1]:
            host_type_idx += ", "
    host_type_idx += "}"
    patch_code = patch_c_code.code_patch_SQWELL.format(
        patch_locations=job._project.generate_patch_location_c_code(job),
        # job.doc.patch_locations is an N_host_types x 3 x 3 array, so take
        # length of the first item for number of patches
        n_patches=len(job.doc.patch_locations[0]),
        sigma=job.sp.sigma,
        lambdasigma=job.sp.lambdasigma,
        host_type_idx=host_type_idx,
    )
    print(patch_code)
    patches = hoomd.jit.patch.user(mc,
                                   code=patch_code,
                                   r_cut=2.0,
                                   array_size=1)
    patches.alpha_iso[0] = 1 / job.sp.kT_ramp[0]

    # run
    while True:
        if hoomd.get_step() > final_step:
            return
        try:
            # set new temperature
            new_kT = calculate_temp(job, hoomd.get_step())
            patches.alpha_iso[0] = 1 / new_kT

            # tune particle moves if kT has changed more than 0.01
            do_tuning = abs(last_tuning_kT - new_kT) > 0.01
            if do_tuning:
                for tune_block in range(n_tune_blocks):
                    for ptype in system.particles.types:
                        mc.shape_param[ptype].ignore_statistics = False
                        for other_type in system.particles.types:
                            if other_type != ptype:
                                mc.shape_param[
                                    other_type].ignore_statistics = True
                        hoomd.run(n_tune_steps)
                        mc_tuners[ptype].update()
                # save the deltas in the job document
                if hoomd.comm.get_rank() == 0:
                    for t in system.particles.types:
                        job.doc.mc_d[t] = mc.get_d(t)
                        job.doc.mc_a[t] = mc.get_a(t)
                    job.doc.last_tuning_kT = new_kT
                hoomd.comm.barrier()
                last_tuning_kT = new_kT

            # stop ignoring statistics from tuning
            for ptype in system.particles.types:
                mc.shape_param[ptype].ignore_statistics = False

            # run
            hoomd.run(n_run_steps, limit_multiple=restart_frequency)
            if hoomd.comm.get_rank() == 0:
                job.doc.timestep = hoomd.get_step()
            hoomd.comm.barrier()
            restart_writer.write_restart()
        except hoomd.WalltimeLimitReached:
            restart_writer.write_restart()
            if hoomd.comm.get_rank() == 0:
                job.doc.timestep = hoomd.get_step()
            hoomd.comm.barrier()
            return
    return


def dict_product(dd):
    keys = dd.keys()
    for element in itertools.product(*dd.values()):
        yield dict(zip(keys, element))


if __name__ == '__main__':
    pr = Project()
    output_dir = os.path.join(pr.root_directory(), 'output-files')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    pr.main()
