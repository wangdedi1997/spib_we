#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

mkdir -pv /tmp/$SLURM_JOB_ID/$WEST_CURRENT_SEG_ID
cd /tmp/$SLURM_JOB_ID/$WEST_CURRENT_SEG_ID

ln -sv $WEST_SIM_ROOT/common_files/plumed_aladip.dat .
ln -sv $WEST_SIM_ROOT/common_files/aladip.pdb .

if [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_CONTINUES" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/aladip_openmm.py > aladip_openmm.py
  ln -sv $WEST_PARENT_DATA_REF/seg.xml ./parent.xml
elif [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_NEWTRAJ" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/aladip_openmm.py > aladip_openmm.py
  ln -sv $WEST_PARENT_DATA_REF ./parent.xml
fi

# Run the dynamics with OpenMM
python aladip_openmm.py

# Obtain data with Plumed
python $WEST_SIM_ROOT/common_files/get_pcoord.py
cat pcoord.dat > $WEST_PCOORD_RETURN
cat label.dat > $WEST_LABEL_RETURN

# Clean up
rm -f aladip* *.dat COLVAR

cd $WEST_SIM_ROOT
mkdir -pv $WEST_CURRENT_SEG_DATA_REF
cd $WEST_CURRENT_SEG_DATA_REF
cp -R /tmp/$SLURM_JOB_ID/$WEST_CURRENT_SEG_ID/. $WEST_CURRENT_SEG_DATA_REF

rm -R /tmp/$SLURM_JOB_ID/$WEST_CURRENT_SEG_ID