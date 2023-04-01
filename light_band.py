import sys

sys.path.insert(0, '/Users/pschwendy/meep-1.25.0/python')
import math
import meep as mp
from meep import mpb
from meep.materials import cSi

num_bands = 8
#geometry = [mp.Cylinder(0.2, material=mp.Medium(epsilon=12), center=mp.Vector3(1/3,1/3)), mp.Cylinder(0.2, material=mp.Medium(epsilon=12), center=mp.Vector3(2/3,2/3))]
# form triangles 
# FORM concave double triangle 
# draw out with matplotlib and draw geometry lattice
vertices = [mp.Vector3(1.675, -0.325 * math.sqrt(3)),
            mp.Vector3(0.675, 0.675 * math.sqrt(3)),
            mp.Vector3(0.325, 0.325 * math.sqrt(3)),
            mp.Vector3(-1.675,0.325 * math.sqrt(3)),
            mp.Vector3(-0.675, -0.675 * math.sqrt(3)),
            mp.Vector3(-0.325,-0.325 * math.sqrt(3))]

geometry = [mp.Prism(vertices, height=1.5, center=mp.Vector3(), material=cSi)]

resolution = 32


a = 4/3  # distance between nearest neighbors
geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1),
                                basis1=(0.5 * a * mp.Vector3(3, math.sqrt(3))),
                                basis2=(0.5 * a * mp.Vector3(3, -math.sqrt(3))))

k_points = [mp.Vector3(),              # Gamma
              mp.Vector3(y=0.5),          # M
              mp.Vector3(-1 / 3, 1 / 3),  # K
              mp.Vector3()]               # Gamma

k_points = mp.interpolate(4, k_points)

ms = mpb.ModeSolver(num_bands=num_bands,
                    k_points=k_points,
                    geometry=geometry,
                    geometry_lattice=geometry_lattice,
                    resolution=resolution)

ms.run_tm()
tm_freqs = ms.all_freqs
tm_gaps = ms.gap_list

# write tm_freqs
with open('tm_freqs.txt', 'w') as f_tm:
    for freq in tm_freqs:
        for i in freq:
            f_tm.write(str(i) + ' ')
        f_tm.write('\n')

# write tm_gaps
with open('tm_gaps.txt', 'w') as f_tm_gaps:
    
    for gap in tm_freqs:
        for i in gap:
            f_tm_gaps.write(str(i) + ' ')
        f_tm_gaps.write('\n')

ms.run_te()
te_freqs = ms.all_freqs
te_gaps = ms.all_freqs
# write te_freqs
with open('te_freqs.txt', 'w') as f_te:
    for freq in te_freqs:
        for i in freq:
            f_te.write(str(i) + ' ')
        f_te.write('\n')

# write te_gaps
with open('te_gaps.txt', 'w') as f_te_gaps:
    for gap in te_gaps:
        for i in gap:
            f_te_gaps.write(str(i) + ' ')
        f_te_gaps.write('\n')