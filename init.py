import itertools
import numpy as np

from project import Project
from project import dict_product

project = Project()
guest_info = [[(0.25, 0.3)]]
parameters = {
    'phi': [0.6],
    'initial_config': ['dilute'],
    'replica': [0, 1, 2, 3, 4],
    'n_repeats': [[40, 40]],
    'betaP': [None],
    'use_floppy_box': [False],
    'patch_offset': [(0.35, 0.5, 0.75)],
    'n_host_edges': [3],
    'n_guest_edges': [4],
    'sigma': [0.0],
    'lambdasigma': [0.1],
    'kT_ramp': [(0.25, 0.05, int(1e7))],
    'guest_info': guest_info,
}
for sp in dict_product(parameters):
    job = project.open_job(sp).init()
    if 'initialize' in [x.name for x in project.next_operations(job)]:
        project.run([job], ['initialize'])
    print('Added job with id: {}'.format(job._id))
