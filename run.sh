#!/bin/bash
#PBS -N gpt2
#PBS -l select=1:ncpus=1:mem=20gb:scratch_local=100gb:ngpus=1:gpu_cap=compute_70:gpu_mem=16gb
#PBS -q gpu
#PBS -l walltime=18:00:00
#PBS -M namlh201@gmail.com

HOMEDIR=/storage/plzen1/home/namlh201
ENVDIR=$HOMEDIR/miniconda3/envs/npfl087

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR=$HOMEDIR/npfl087

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# if the copy operation fails, issue error message and exit
cp -r $DATADIR $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }


cd $SCRATCHDIR/npfl087
$ENVDIR/bin/python main.py

JOB_ID=$(echo ${PBS_JOBID:0:8})

# copy back to DATADIR
mkdir $DATADIR/$JOB_ID
cp -r $SCRATCHDIR/npfl087/models $DATADIR/$JOB_ID

# clean the SCRATCH directory
clean_scratch%