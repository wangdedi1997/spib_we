from sys import stdout
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from openmmplumed import PlumedForce
import mdtraj as md

pdb = PDBFile('aladip.pdb')
topology = md.load('aladip.pdb').topology

forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')

# The modeller builds a periodic box with the solute and solvent molecules.
# PME is the method to compute long-range electristatic interactions in
# periodic systems.
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, constraints=HBonds)

# create plumed force
with open('plumed_aladip.dat',mode='r') as f:
    script = f.read()
force = PlumedForce(script)
system.addForce(force)

temperature = 300 * kelvin
pressure = 1 * bar
    
integrator = LangevinIntegrator(temperature, 1/picosecond, 2*femtoseconds)
integrator.setRandomNumberSeed(RAND)

system.addForce(MonteCarloBarostat(pressure, temperature))
simulation = Simulation(pdb.topology, system, integrator)
simulation.loadState('parent.xml')

# reset the current time of the simulation
simulation.context.setTime(0)
simulation.context.setStepCount(0)

simulation.reporters.append(md.reporters.DCDReporter('md.dcd', 5000, atomSubset=topology.select('protein')))
simulation.reporters.append(StateDataReporter(stdout, 5000, step=True, potentialEnergy=True, temperature=True, elapsedTime=True))
simulation.step(10000)
simulation.saveState('seg.xml')
