from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import pickle as pk
import shutil
import os
import pdbfixer

on_gpu = True

path = os.getcwd()

def fix_pdb(raw_pdb):
    """
    fixes the raw pdb from colabfold using pdbfixer.
    This needs to be performed to cleanup the pdb and to start simulation 
    Fixes performed: missing residues, missing atoms and missing Terminals
    """
    # fixer instance
    fixer = pdbfixer.PDBFixer(raw_pdb)
    #finding and adding missing residues including terminals
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)

    """
    Adds missing hydrogen to the pdb for a particular forcefield
    """
    forcefield = ForceField(path+'/protein.ff14SBonlysc.xml', 'implicit/gbn2.xml')
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield)

    out_handle = open(raw_pdb[:-4]+'_fixed.pdb','w')
    PDBFile.writeFile(modeller.topology, modeller.positions, out_handle, keepIds=True)



def prepare_chignolin():
    """
    Prepares the chignolin system for Molecular Dynamics (MD) simulations.
    Downloads the pdb structure from
    http://ambermd.org/tutorials/advanced/tutorial22/files/5PTI-DtoH-dry.pdb
    and parameterizes it using General Amber Force Field
    (GAFF).

    """
    os.system("curl -O https://files.rcsb.org/download/2RVD.pdb1.gz")
    os.system("gunzip 2RVD.pdb1.gz")
    os.system("mv 2RVD.pdb1 chignolin.pdb")

    os.system("grep -e '^ATOM\|^HETATM\|^TER\|^END' chignolin.pdb > system.pdb")
    os.system("rm -rf system_inputs")
    os.system("mkdir system_inputs")
    cwd = os.getcwd()
    target_dir = cwd + "/" + "system_inputs"

    fix_pdb("system.pdb")
    shutil.copy(
        cwd + "/" + "chignolin.pdb", target_dir + "/" + "chignolin.pdb"
    )
    shutil.copy(
        cwd + "/" + "system.pdb", target_dir + "/" + "system.pdb"
    )
    shutil.copy(
        cwd + "/" + "system.pdb", target_dir + "/" + "system_fixed.pdb"
    )
    os.system("rm -rf chignolin.pdb")
    os.system("rm -rf system.pdb")
    os.system("rm -rf system_fixed.pdb")


def simulated_annealing(
    input_pdb='system_fixed.pdb',
    annealing_output_pdb="system_annealing_output.pdb",
    annealing_steps=1000,
    pdb_freq=5000,
    starting_temp=0,
    target_temp=340,
    temp_incr=5,
):

    """

    Performs simulated annealing of the system from
    0K to 340 K (default) using OpenMM MD engine and
    saves the last frame of the simulation to be
    accessed by the next simulation.

    Parameters
    ----------

    annealing_output_pdb: str
        System's output trajectory file

    annealing_steps: int
        Aneealing steps at each temperatrure jump

    pdb_freq: int
        Trajectory to be saved after every pdb_freq steps

    starting_temp: int
        Initial temperature of Simulated Annealing

    target_temp: int
        Final temperature of Simulated Annealing

    temp_incr: int
        Temmperature increase for every step

    """

    forcefield = ForceField(path+'/protein.ff14SBonlysc.xml', 'implicit/gbn2.xml')
    pdb = PDBFile(input_pdb)
    annealing_system = forcefield.createSystem(pdb.topology, constraints=HBonds)
    annealing_integrator = LangevinIntegrator(
        0 * kelvin, 1 / picosecond, 2 * femtoseconds
    )
    total_steps = ((target_temp / temp_incr) + 1) * annealing_steps
    annealing_simulation = Simulation(
        pdb.topology,
        annealing_system,
        annealing_integrator
    )

    annealing_simulation.context.setPositions(pdb.positions)
    annealing_simulation.minimizeEnergy()
    annealing_simulation.reporters.append(
        PDBReporter(annealing_output_pdb, pdb_freq)
    )
    simulated_annealing_last_frame = (
        annealing_output_pdb[:-4] + "_last_frame.pdb"
    )
    annealing_simulation.reporters.append(
        PDBReporter(simulated_annealing_last_frame, total_steps)
    )
    annealing_simulation.reporters.append(
        StateDataReporter(
            stdout,
            pdb_freq,
            step=True,
            time=True,
            potentialEnergy=True,
            totalSteps=total_steps,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            separator="\t",
        )
    )
    temp = starting_temp
    while temp <= target_temp:
        annealing_integrator.setTemperature(temp * kelvin)
        annealing_simulation.step(annealing_steps)
        temp += temp_incr
    state = annealing_simulation.context.getState()
    print(state.getPeriodicBoxVectors())
    annealing_simulation_box_vectors = state.getPeriodicBoxVectors()
    print(annealing_simulation_box_vectors)
    with open("annealing_simulation_box_vectors.pkl", "wb") as f:
        pk.dump(annealing_simulation_box_vectors, f)
    print("Finshed NVT Simulated Annealing Simulation")
    annealing_simulation.saveState("annealing.state")


def nvt_equilibration(
    nvt_output_pdb="system_nvt_output.pdb",
    pdb_freq=500000,
    nvt_steps=5000000,
    target_temp=340,
    nvt_pdb="system_annealing_output_last_frame.pdb",
):

    """

    Performs NVT equilibration MD of the system
    using OpenMM MD engine  saves the last
    frame of the simulation to be accessed by
    the next simulation.

    Parameters
    ----------
    parm: str
        System's topology file

    nvt_output_pdb: str
        System's output trajectory file

    pdb_freq: int
        Trajectory to be saved after every pdb_freq steps

    nvt_steps: int
        NVT simulation steps

    target_temp: int
        Temperature for MD simulation

    nvt_pdb: str
        Last frame of the simulation

    """

    nvt_init_pdb = PDBFile(nvt_pdb)
    forcefield = ForceField(path+'/protein.ff14SBonlysc.xml', 'implicit/gbn2.xml')
    nvt_system = forcefield.createSystem(nvt_init_pdb.topology, constraints=HBonds)
    nvt_integrator = LangevinIntegrator(
        target_temp * kelvin, 1 / picosecond, 2 * femtoseconds
    )
    nvt_simulation = Simulation(
        nvt_init_pdb.topology,
        nvt_system,
        nvt_integrator
    )
    nvt_simulation.context.setPositions(nvt_init_pdb.positions)
    nvt_simulation.context.setVelocitiesToTemperature(target_temp * kelvin)
    nvt_last_frame = nvt_output_pdb[:-4] + "_last_frame.pdb"
    nvt_simulation.reporters.append(PDBReporter(nvt_output_pdb, pdb_freq))
    nvt_simulation.reporters.append(PDBReporter(nvt_last_frame, nvt_steps))
    nvt_simulation.reporters.append(
        StateDataReporter(
            stdout,
            pdb_freq,
            step=True,
            time=True,
            potentialEnergy=True,
            totalSteps=nvt_steps,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            separator="\t",
        )
    )
    nvt_simulation.minimizeEnergy()
    nvt_simulation.step(nvt_steps)
    nvt_simulation.saveState("nvt_simulation.state")
    state = nvt_simulation.context.getState()
    print(state.getPeriodicBoxVectors())
    nvt_simulation_box_vectors = state.getPeriodicBoxVectors()
    print(nvt_simulation_box_vectors)
    with open("nvt_simulation_box_vectors.pkl", "wb") as f:
        pk.dump(nvt_simulation_box_vectors, f)
    print("Finished NVT Simulation")


def run_equilibration():

    """

    Runs systematic simulated annealing followed by
    NVT equilibration MD simulation.

    """

    cwd = os.getcwd()
    target_dir = cwd + "/" + "equilibration"
    os.system("rm -rf equilibration")
    os.system("mkdir equilibration")
    
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "chignolin.pdb",
        target_dir + "/" + "chignolin.pdb",
    )
    shutil.copy(
        cwd + "/" + "system_inputs" + "/" + "system_fixed.pdb",
        target_dir + "/" + "system_fixed.pdb",
    )
    os.chdir(target_dir)
    simulated_annealing()
    nvt_equilibration()
    os.system("rm -rf chignolin.pdb")
    os.system("rm -rf system.pdb")
    os.chdir(cwd)

prepare_chignolin()
run_equilibration()
