from ast import Num
from math import pi
from unittest.mock import patch
import numpy as np
import rowan
import signac

def calculate_fitting_space(host_verticies, patch_locations) -> Num:
    modified_patches = np.array(patch_locations)
    for i in range(4):
        a = modified_patches[2]
        b = modified_patches[2] + modified_patches[2] - modified_patches[0]
        c = modified_patches[2] + modified_patches[2] - modified_patches[1]
        modified_patches = np.array([a, b, c])

    return min([x for x,y,z in modified_patches]) - max([x for x,y,z in host_verticies])

def total_fit(job, fitting_space, particle_id) -> Num:
    guest_vertices = np.array(job.doc.guest_vertices[particle_id])
    total_fit = 0

    for d_theta in range(100):
        theta = (d_theta / 100) * (2 * pi)
        q = rowan.from_axis_angle([4, 3, 0], theta)
        new_pos = rowan.rotate(q, guest_vertices)
        total_fit += max(fitting_space - abs(new_pos[0][0] - new_pos[2][0]), 0)
    
    return total_fit

def main():
    project = signac.get_project('../../aug-self-assembly/')
    job = project.open_job(id='f39ec8bce')
    
    host_vertices = job.doc.host_vertices
    patch_locations = job.doc.patch_locations[0]

    fitting_space = calculate_fitting_space(host_vertices, patch_locations)
    
    total_fit_B = total_fit(job, fitting_space, 'B')
    total_fit_C = total_fit(job, fitting_space, 'C')
    print(f'{job.sp.guest_info[0][0]}=>{total_fit_B}')
    print(f'{job.sp.guest_info[1][0]}=>{total_fit_C}')


if __name__ == '__main__':
    main()