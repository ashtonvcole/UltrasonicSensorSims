from math import pi, sin, cos



###########################
# USER-DEFINED PARAMETERS #
###########################

R = 0.5 # Radius
L = 10 # Length
theta = 360 * pi / 180 # wedge angle
nbth = 4 # Number of blocks along theta, each at most pi / 2
nbz = 1 # Number of blocks along z-axis
ndpbth = 36 # Number of angular divisions per block
ndpbr = 40 # Number of divisions along radius per block
ndpbz = 800 # Number of divisions along z-axis per block
grading = 0.025 # Grading for boundary layer along radius
case = 'cfd' # 'cfd' or 'acoustic'
interior = 'symmetry' # 'symmetry', 'cyclic', or 'wedge'
c2m = 1.0 # Conversion to meters
fname = 'blockMeshDict.fcyl10-c'



###################
# MESH GENERATION #
###################

assert R > 0
assert L > 0
assert theta > 0 and theta <= 2 * pi
assert nbth > 0 and nbth * pi >= 2 * theta # Block th <= pi / 2
assert nbz > 0
assert ndpbth > 0
assert ndpbr > 0

full = abs(theta - 2 * pi) < 0.001
if (full):
    print('Generating full cylinder')

diff = nbth + 2 if (not full) else nbth + 1 # Difference in indexing
                                            # between two vertices with the same r and theta
                                            # offset by one block

f = open(fname, 'w+')



# File information

f.write('FoamFile\n{\n')
f.write('\tversion 2.0;\n')
f.write('\tformat ascii;\n')
f.write('\tclass dictionary;\n')
f.write('\tobject blockMeshDict;\n')
f.write('}\n')
f.write(f'\nconvertToMeters {c2m};\n')



# Vertices

f.write('\nvertices\n(')

vi = 0
for ci in range(0, nbz + 1):
    f.write(f'\n\t// Cross-section {ci}\n')
    # Center vertex
    f.write(f'\t({0:e}\t{0:e}\t{ci / nbz * L:e})\t// {vi}\n')
    vi += 1
    for n in range(0, nbth):
        # Outer vertices
        f.write(f'\t({R * cos(n / nbth * theta):e}\t{R * sin(n / nbth * theta):e}\t{ci / nbz * L:e})\t// {vi}\n')
        vi += 1
    if (not full):
        # Additional vertex if not full cylinder
        f.write(f'\t({R * cos(theta):e}\t{R * sin(theta):e}\t{ci / nbz * L:e})\t// {vi}\n')
        vi += 1
    else:
        f.write(f'\t// Full circle\n')

f.write(');\n')



# Blocks

f.write('\nblocks\n(')

for ci in range(0, nbz):
    f.write(f'\n\t// Blocks section {ci}, i.e. cross-sections {ci} to {ci + 1}\n')
    for n in range(0, nbth - 1):
        f.write(f'\thex ({ci * diff} {(ci + 1) * diff} {(ci + 1) * diff + n + 1} {ci * diff + n + 1} {ci * diff} {(ci + 1) * diff} {(ci + 1) * diff + n + 2} {ci * diff + n + 2})')
        f.write(f' ({ndpbz} {ndpbr} {ndpbth})')
        f.write(f' simpleGrading (1.0 {grading} 1.0)')
        f.write(f' // Block {ci * nbth + n}\n')
    if (not full):
        f.write(f'\thex ({ci * diff} {(ci + 1) * diff} {(ci + 1) * diff + nbth} {ci * diff + nbth} {ci * diff} {(ci + 1) * diff} {(ci + 1) * diff + nbth + 1} {ci * diff + nbth + 1})')
        f.write(f' ({ndpbz} {ndpbr} {ndpbth})')
        f.write(f' simpleGrading (1.0 {grading} 1.0)')
        f.write(f' // Block {(ci + 1) * nbth - 1}\n')
    else:
        f.write(f'\thex ({ci * diff} {(ci + 1) * diff} {(ci + 1) * diff + nbth} {ci * diff + nbth} {ci * diff} {(ci + 1) * diff} {(ci + 1) * diff + 1} {ci * diff + 1})')
        f.write(f' ({ndpbz} {ndpbr} {ndpbth})')
        f.write(f' simpleGrading (1.0 {grading} 1.0)')
        f.write(f' // Block {(ci + 1) * nbth - 1} (full circle)\n')

f.write(');\n')



# Edges

f.write('\nedges\n(')

vi = 1
for ci in range(0, nbz + 1):
    f.write(f'\n\t// Cross-section {ci}\n')
    for n in range(0, nbth - 1):
        f.write(f'\tarc {vi} {vi + 1} ({R * cos((n + 0.5) / nbth * theta):e} {R * sin((n + 0.5) / nbth * theta):e} {ci / nbz * L:e}) // {vi} to {vi + 1}\n')
        vi += 1
    if (not full):
        f.write(f'\tarc {vi} {vi + 1} ({R * cos((nbth - 0.5) / nbth * theta):e} {R * sin((nbth - 0.5) / nbth * theta):e} {ci / nbz * L:e}) // {vi} to {vi + 1}\n')
        vi += 1
    else:
        f.write(f'\tarc {vi} {ci * diff + 1} ({R * cos((nbth - 0.5) / nbth * theta):e} {R * sin((nbth - 0.5) / nbth * theta):e} {ci / nbz * L:e}) // {vi} to {ci * diff + 1} (full circle)\n')
    vi += 2

f.write(');\n')



# Boundary conditions

f.write('\nboundary\n(\n')

if (case == 'cfd'):
    f.write('\tinlet\n\t{\n\t\ttype patch;\n\t\tfaces\n\t\t(\n')
    for n in range(0, nbth - 1):
        f.write(f'\t\t\t({0} {n + 2} {n + 1} {0})\n')
    if (not full):
        f.write(f'\t\t\t({0} {nbth + 1} {nbth} {0})\n')
    else:
        f.write(f'\t\t\t({0} {1} {nbth} {0})\n')
    f.write('\t\t);\n\t}\n')
    f.write('\n\toutlet\n\t{\n\t\ttype patch;\n\t\tfaces\n\t\t(\n')
    for n in range(0, nbth - 1):
        f.write(f'\t\t\t({nbz * diff} {nbz * diff + n + 1} {nbz * diff + n + 2} {nbz * diff})\n')
    if (not full):
        f.write(f'\t\t\t({nbz * diff} {nbz * diff + nbth} {nbz * diff + nbth + 1} {nbz * diff})\n')
    else:
        f.write(f'\t\t\t({nbz * diff} {nbz * diff + nbth} {nbz * diff + 1} {nbz * diff})\n')
    f.write('\t\t);\n\t}\n')
    f.write('\n\twall\n\t{\n\t\ttype wall;\n\t\tfaces\n\t\t(')
    for ci in range(0, nbz):
        f.write(f'\n\t\t\t// Blocks section {ci}\n')
        for n in range(0, nbth - 1):
            f.write(f'\t\t\t({ci * diff + n + 1} {ci * diff + n + 2} {(ci + 1) * diff + n + 2} {(ci + 1) * diff + n + 1}) // Block {ci * nbth + n}\n')
        if (not full):
            f.write(f'\t\t\t({ci * diff + nbth} {ci * diff + nbth + 1} {(ci + 1) * diff + nbth + 1} {(ci + 1) * diff + nbth}) // Block {(ci + 1) * nbth - 1}\n')
        else:
            f.write(f'\t\t\t({ci * diff + nbth} {ci * diff + 1} {(ci + 1) * diff + 1} {(ci + 1) * diff + nbth}) // Block {(ci + 1) * nbth - 1} (full circle)\n')
    f.write('\t\t);\n\t}\n')
    if (not full):
        f.write('\n\tinteriorA\n\t{\n')
        if (interior == 'symmetry'):
            f.write('\t\ttype symmetryPlane;\n')
        elif (interior == 'cyclic'):
            f.write('\t\ttype cyclic;\n\t\tneighbourPatch interiorB;\n')
        elif (interior == 'wedge'):
            print('WARNING: interior wedge boundaries not yet defined, blockMesh boundary patch generation incomplete')
        else:
            print('WARNING: unsupported treatment of interior faces, blockMesh boundary patch generation incomplete')
        f.write('\t\tfaces\n\t\t(\n')
        for ci in range(0, nbz):
            f.write(f'\t\t\t({ci * diff} {ci * diff + 1} {(ci + 1) * diff + 1} {(ci + 1) * diff})\n')
        f.write('\t\t);\n\t}\n')
        f.write('\n\tinteriorB\n\t{\n')
        if (interior == 'symmetry'):
            f.write('\t\ttype symmetryPlane;\n')
        elif (interior == 'cyclic'):
            f.write('\t\ttype cyclic;\n\t\tneighbourPatch interiorB;\n')
        elif (interior == 'wedge'):
            print('WARNING: interior wedge boundaries not yet defined, blockMesh boundary patch generation incomplete')
        else:
            print('WARNING: unsupported treatment of interior faces, blockMesh boundary patch generation incomplete')
        f.write('\t\tfaces\n\t\t(\n')
        for ci in range(0, nbz):
            f.write(f'\t\t\t({ci * diff} {(ci + 1) * diff} {(ci + 2) * diff - 1} {(ci + 1) * diff - 1})\n')
        f.write('\t\t);\n\t}\n')
elif (case == 'acoustic'):
    print('WARNING: acoustic case boundaries not yet defined, blockMesh boundary patch generation incomplete')
else:
    print('WARNING: unsupported case, blockMesh boundary patch generation incomplete')

f.write(');\n')

f.close()

