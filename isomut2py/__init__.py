# command line parameter "-d" for samtools, used for all pileup file generation
SAMTOOLS_MAX_DEPTH=1000
# path to package
HOME = __path__[0]

__version__ = '2.0.1'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from isomut2py import mutationcalling
from isomut2py import ploidyestimation
from isomut2py import bayesian
from isomut2py import compare
from isomut2py import format
from isomut2py import io
from isomut2py import plot
from isomut2py import postprocess
from isomut2py import process
from isomut2py import examples

import subprocess as __subprocess
import os as __os

# warnings.resetwarnings()

# compiling C scripts
for __fname in __os.listdir(HOME+'/C'):
    if __fname.split('.')[-1] != 'h' and __fname.split('.')[-1] != 'c':
        __subprocess.call('rm ' + HOME + '/C/' + __fname, shell=True)

print('Compiling C scripts...')
cwd = __os.getcwd()
__compiled = []
__os.chdir(HOME + '/C')
__cmd = 'gcc -c -O3 isomut2_lib.c fisher.c  -W -Wall'
__compiled.append(__subprocess.call(__cmd, shell=True))
__cmd = 'gcc -O3 -o isomut2 isomut2.c isomut2_lib.o fisher.o -lm -W -Wall'
__compiled.append(__subprocess.call(__cmd, shell=True))
__cmd = 'gcc -O3 -o PEprep PEprep.c isomut2_lib.o fisher.o -lm -W -Wall'
__compiled.append(__subprocess.call(__cmd, shell=True))
__cmd = 'gcc -O3 -o checkPileup checkPileup.c isomut2_lib.o fisher.o -lm -W -Wall'
__compiled.append(__subprocess.call(__cmd, shell=True))
__os.chdir(cwd)

if 1 in __compiled:
    print('WARNING: C scripts were not compiled correctly, some functions will not work.')
    print('Try compiling manually to ' + HOME + '/C')
else:
    print('Done.')