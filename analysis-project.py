import collections
import flow
from flow import FlowProject, directives, cmd
from flow import environments
import fresnel
import freud
import gsd.hoomd
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from project import calculate_temp, done_running
import rowan
import scipy.spatial


class AnalysisProject(FlowProject):
    def __init__(self):
        flow.FlowProject.__init__(self)


def generic_post_condition(job, fn_tail):
    ts_current = job.doc.get('timestep', 0)
    if not job.isfile(fn_tail):
        return False
    data = np.loadtxt(job.fn(fn_tail), ndmin=1)
    ts_last_plot = data[-1]
    if ts_current > ts_last_plot:
        return False
    else:
        return True

def write_ts_for_post_condition(job, fn_tail):
    with open(job.fn(fn_tail), 'a') as f:
        ts = job.doc.get('timestep', 0)
        f.write(f'{ts:d}\n')
    return


def get_num_particles(job):
    gsd_fn = job.fn('init.gsd')
    num_particles = 0
    with gsd.hoomd.open(gsd_fn, 'rb') as traj:
        num_particles = traj[0].particles.N
    return num_particles


def clean_energy_data(data, skip=100):
    """Remove values logged as 0.0 from restarts

    Args
    ----
    data : np.ndarray, shape=(n,)
        The data to 'clean'
    skip : int
        Number of items to ignore at beginning of array

    Returns
    -------
    clean_data : np.ndarray, shape=(n,)
        The cleaned data, with 0.0 values replaced with the previous value

    """
    zero_locs = np.argwhere(data == 0.0)
    # change where zero_locs == 0 to 1 so that when we do zero_locs-1 we don't
    # index the last element
    zero_locs[zero_locs==0] = 1
    data[zero_locs] = data[zero_locs-1]
    return data


@AnalysisProject.operation
@AnalysisProject.pre.isfile('hpmc-patch-stats.txt')
@AnalysisProject.post(lambda job: 
        generic_post_condition(job, 'energy-plot-timesteps.txt'))
def plot_energy_timeseries(job):
    fn = job.fn('hpmc-patch-stats.txt')
    data = np.genfromtxt(fn, names=True)
    fig, ax = plt.subplots()
    ts = data['timestep']
    pe = data['hpmc_patch_energy']
    pe = clean_energy_data(pe)
    kT = np.empty_like(ts)
    for idx, ts_ in enumerate(ts):
        kT[idx] = calculate_temp(job, ts_)
    pe *= kT  # give it energy units
    pe /= get_num_particles(job)  # make it intensive
    ll, = ax.plot(ts, pe)
    ax.set_title(f'{job._id[:6]}')
    ax.set_xlabel('HPMC step')
    ax.set_ylabel(r'$U/N$', c=ll.get_c())
    ax.tick_params(colors=ll.get_c(), which='both', axis='y')
    ax2 = ax.twinx()
    ax2.plot([], [])  # just to increment color cycler
    ll, = ax2.plot(ts, kT)
    ax2.set_ylabel('kT', c=ll.get_c())
    ax2.tick_params(colors=ll.get_c(), which='both', axis='y')
    fig.set_facecolor('w')
    figfn = job.fn('pe.png')
    fig.tight_layout()
    fig.savefig(figfn, transparent=False)
    plt.close(fig)
    write_ts_for_post_condition(job, 'energy-plot-timesteps.txt')
    return

def get_frame(filename, index):
    with gsd.hoomd.open(filename, mode='rb') as traj:
        frame = traj[index]
    return frame

@AnalysisProject.operation
@AnalysisProject.pre.isfile('restart.gsd')
@AnalysisProject.post(lambda j: generic_post_condition(j, 'snapshot-timesteps.txt'))
def plain_snapshot(job):
    frame = get_frame(job.fn('restart.gsd'), -1)
    device = fresnel.Device()
    scene = fresnel.Scene(device=device)
    start_idx = 0
    for idx, n_verts in enumerate(frame.state['hpmc/convex_polygon/N']):
        end_idx = start_idx + n_verts        
        verts = np.array(
            frame.state['hpmc/convex_polygon/vertices'][start_idx:end_idx])
        this_type = frame.particles.typeid == idx
        num_this_type = np.count_nonzero(this_type)
        geometry = fresnel.geometry.Polygon(
            scene, N=num_this_type, vertices=verts)
        geometry.position[:] = frame.particles.position[this_type, :2]
        ors = frame.particles.orientation[this_type]
        particle_diameter = 2 * np.amax(np.linalg.norm(verts, axis=1))
        geometry.angle[:] = 2 * np.arctan2(ors[:, 3], ors[:, 0])
        geometry.material.solid = 1.0
        color = np.array(plt.matplotlib.colors.to_rgb(f'C{idx}'))
        scene.background_color = (1.0, 1.0, 1.0)
        scene.background_alpha = 1.0
        geometry.material.color = fresnel.color.linear(color)
        start_idx = end_idx
    scene.camera = fresnel.camera.Orthographic.fit(scene)
    tracer = fresnel.tracer.Preview(device=device, w=600, h=600)
    out = tracer.render(scene)
    im = PIL.Image.fromarray(out[:], mode='RGBA')
    fn = job.fn('snapshot.png')
    im.save(fn)
    write_ts_for_post_condition(job, 'snapshot-timesteps.txt')
    return


@AnalysisProject.operation
@AnalysisProject.pre.isfile('traj.gsd')
@AnalysisProject.pre(lambda j: done_running(j))
@AnalysisProject.post(lambda j: generic_post_condition(j, 'pore-analysis-timesteps.txt'))
@directives(
    walltime=0.5,
    np=4,
)
def analyze_pore_loading(job):
    import host_guest_analysis as hga
    filename = job.fn('traj.gsd')
    if not 'pore_stats' in job.doc:
        job.doc.pore_stats = dict()
    num_guest_types = len(job.sp.guest_info)
    with gsd.hoomd.open(filename, 'rb') as gsd_file:
        for frame in gsd_file:
            timestep = frame.configuration.step
            if str(timestep) in job.doc.pore_stats:
                continue

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
                frame.particles.typeid != typeid_A).flatten()
            host_positions = pos[host_idxs]
            host_orientations = ors[host_idxs]
            guest_positions = pos[guest_idxs]
            patch_range = job.sp.lambdasigma
            patch_locations = job.doc.patch_locations[0]

            # create host-host neighborlist and adjacency matrix
            host_host_nlist = hga.get_patch_nlist(
                box,
                patch_locations,
                patch_range,
                host_positions,
                host_orientations,
            )
            adj_matrix = hga.create_adjacency_matrix(
                host_host_nlist,
                len(host_positions),
            )

            # find hexamers from adjacency matrix
            hexamers = hga.find_hexamers(adj_matrix, len(host_idxs))
            hosts_in_hexamers = np.unique(
                np.array(hexamers, dtype=np.int64).flatten())
            global_idxs_hosts_in_hexamers = host_idxs[hosts_in_hexamers]

            # find pore centers
            _, indices = np.unique(
                np.sort(np.array(hexamers)),
                axis=0,
                return_index=True,
            )
            unique_hexamers = np.array(hexamers, dtype=np.int64)[indices]
            pore_centers, search_radius = hga.get_pore_centers(
                unique_hexamers,
                host_positions,
                box,
            )

            # find captured guests
            if len(hexamers) > 0:
                captured_guest_idxs = hga.find_guests_in_pores(
                    guest_positions,
                    pore_centers,
                    search_radius,
                    box,
                )
            else:
                captured_guest_idxs = []
            global_idxs_guests_in_pores = guest_idxs[captured_guest_idxs]
            num_pores = len(unique_hexamers)
            num_captured_guests = len(captured_guest_idxs)
            cc = collections.Counter(frame.particles.typeid[global_idxs_guests_in_pores])
            frame_stats = [num_pores]
            frame_stats.extend(
                [cc[gtypid] for gtypid in range(1, 1 + num_guest_types)]
            )
            job.doc.pore_stats[str(timestep)] = frame_stats
    write_ts_for_post_condition(job, 'pore-analysis-timesteps.txt')

    
if __name__ == '__main__':
    pr = AnalysisProject()
    output_dir = os.path.join(pr.root_directory(), 'output-files')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    pr.main()
