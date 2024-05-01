from sys import stdout
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import os

# sysmtem preparation
def prepare_alanine_dipeptide():
    os.system(
        "curl -O http://ftp.imp.fu-berlin.de/pub/cmb-data/alanine-dipeptide-nowater.pdb"
    )
    os.system("rm -rf system_inputs")
    os.system("mkdir system_inputs")
    
    os.system("mv alanine-dipeptide-nowater.pdb system_inputs/alanine-dipeptide.pdb")
    
    
    pdb = PDBFile('system_inputs/alanine-dipeptide.pdb')
    modeller = Modeller(pdb.topology, pdb.positions)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller.addSolvent(forcefield, model='tip3p', padding=1*nanometer)
    print(modeller.topology)
    # Write a PDB file to provide a topology of the solvated
    # system to MDTraj below.
    with open('system_inputs/sysmtem_preparation.pdb', 'w') as outfile:
        PDBFile.writeFile(modeller.topology, modeller.positions, outfile)
        
# NVT equilibriration
def nvt_equilibration():
    os.system("rm -rf nvt_equilibration")
    os.system("mkdir nvt_equilibration")
    
    pdb = PDBFile('system_inputs/sysmtem_preparation.pdb')
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')

    # The modeller builds a periodic box with the solute and solvent molecules.
    # PME is the method to compute long-range electristatic interactions in
    # periodic systems.
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, constraints=HBonds)

    temperature = 300 * kelvin

    nvt_steps=5000000

    integrator = LangevinIntegrator(temperature, 1/picosecond, 2*femtoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.minimizeEnergy()
    simulation.reporters.append(DCDReporter('nvt_equilibration/nvt.dcd', 5000))
    simulation.reporters.append(StateDataReporter(stdout, 5000, step=True,
            temperature=True, elapsedTime=True))
    simulation.reporters.append(StateDataReporter("nvt_equilibration/nvt.csv", 5000, time=True,
        potentialEnergy=True, totalEnergy=True, temperature=True))

    nvt_last_frame = "nvt_equilibration/nvt.pdb"
    simulation.reporters.append(PDBReporter(nvt_last_frame, nvt_steps))

    simulation.step(nvt_steps)
    simulation.saveState("nvt_equilibration/nvt.state")
    
# NPT equilibriration
def npt_equilibration():
    os.system("rm -rf npt_equilibration")
    os.system("mkdir npt_equilibration")
    
    pdb = PDBFile('nvt_equilibration/nvt.pdb')
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')

    # The modeller builds a periodic box with the solute and solvent molecules.
    # PME is the method to compute long-range electristatic interactions in
    # periodic systems.
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, constraints=HBonds)
    
    temperature = 300 * kelvin
    pressure = 1 * bar

    npt_steps=5000000
    
    integrator = LangevinIntegrator(temperature, 1/picosecond, 2*femtoseconds)
    system.addForce(MonteCarloBarostat(pressure, temperature))
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.minimizeEnergy()
    simulation.reporters.append(DCDReporter('npt_equilibration/npt.dcd', 5000))
    simulation.reporters.append(StateDataReporter(stdout, 5000, step=True,
            temperature=True, elapsedTime=True))
    simulation.reporters.append(StateDataReporter("npt_equilibration/npt.csv", 5000, time=True,
        potentialEnergy=True, totalEnergy=True, temperature=True))

    npt_last_frame = "npt_equilibration/npt.pdb"
    simulation.reporters.append(PDBReporter(npt_last_frame, npt_steps))
    
    simulation.step(npt_steps)
    simulation.saveState("npt_equilibration/npt.state")
    
prepare_alanine_dipeptide()
nvt_equilibration()
npt_equilibration()