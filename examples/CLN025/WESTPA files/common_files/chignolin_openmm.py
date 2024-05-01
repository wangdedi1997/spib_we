from sys import stdout
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from openmmplumed import PlumedForce
import os

WE_folder = os.environ.get('WEST_SIM_ROOT')
file_folder = WE_folder + '/common_files/'

pdb = PDBFile('chignolin.pdb')

forcefield = ForceField(file_folder+'protein.ff14SBonlysc.xml', 'implicit/gbn2.xml')

# The modeller builds a periodic box with the solute and solvent molecules.
# PME is the method to compute long-range electristatic interactions in
# periodic systems.
system = forcefield.createSystem(pdb.topology, constraints=HBonds)

# create plumed force
with open('plumed_chignolin.dat',mode='r') as f:
    script = f.read()
force = PlumedForce(script)
system.addForce(force)

temperature = 340 * kelvin
    
integrator = LangevinIntegrator(temperature, 1/picosecond, 2*femtoseconds)
integrator.setRandomNumberSeed(RAND)

simulation = Simulation(pdb.topology, system, integrator)
simulation.loadState('parent.xml')

# reset the current time of the simulation
simulation.context.setTime(0)
simulation.context.setStepCount(0)

simulation.reporters.append(DCDReporter('md.dcd', 5000))
simulation.reporters.append(StateDataReporter(stdout, 5000, step=True, potentialEnergy=True, temperature=True, elapsedTime=True))
simulation.step(10000)
simulation.saveState('seg.xml')
