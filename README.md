# PyECoG
This modulde is for detecting epileptiform activity from single channel intracranial EEG (or ECoG) recordings.
Currently under heavy constructions.

## Grown out of "networkclassifer"
This was originally a bunch of scripts for classifying brain-network states for acute models. The state in a given timewindow before light pulses that activated neurons. These scripts are still in "lightcode" incase needed.

# Usage:

## Todo:
* package this bad boy up into pip
* currently very ndf file heavy
* gui for visualising traces and storing annotations

## Todo - more minor things
* AR coefs.
* Normalisation options
* Allow easy insertion of custom features
* speed up file conversion
* handle errors better
* make tests
* decide on how to store data in hdf5 files, currently in favour of one file per transmitter_id
* bad message filtering for fs != 512


