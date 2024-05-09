#!/usr/bin/python
import yaml  # type: ignore

"""
This script generates multiple input decks for running an interpolation program on the NCI Gadi supercomputer.
It takes a sample input deck and replaces template fields with user-defined values to create separate input decks for different input variables.
The resulting input decks are saved as individual files with names based on the input variables.

Reference deck: interpolation_nci_gadi.deck
Reference yaml file: user_input_test.yaml
Output decks: runInterp_%s_%s.deck, where %s represents the input variable name and year of interpolation.

To run the generated input decks on NCI Gadi, use the following command: run the first deck
./runInterp_%s_%s.deck # Make sure the deck is executable, e.g., chmod +x runInterp_%s_%s.deck

Interpolation for each input variable over a year can take up to 1 hour with 28 cores and 126 GB memory, using a recent version of the Python package.
The processing time may vary depending on the version of the dask package and the number of cores used.

Contact: Youngil Kim (youngil.kim@unsw.edu.au)
"""

# ======================================================================================
# INPUT
# ======================================================================================

# Name the input deck to use
indeck = "interpolation_nci_gadi.deck"

# Path to the g3 directory. Should start with /
OUT_dir = "/scratch/dm6/yk8692/interpolation/test_v2"

# Path to the input yaml file. Should start with /
yamlfile = "/scratch/dm6/yk8692/sdmbc/user_input_test.yaml"

# Path to the storage
storage = "scratch/w28+scratch/dm6+gdata/w28+gdata/hh5+gdata/oi10+gdata/dm6+gdata/rt52"

# Input variable list
Input_var = ["hus", "ta", "ua", "va"]  # Don't need to change this

# Set the project number to run the programs
project = "dm6"
project2 = "w28"
# Set the project number to save the files.
outproject = "dm6"

# Computing resources
ncores = 96  # number of cores
memory = 378  # in GB
walltime = 2  # in hours

# Email: to receive an email at the end of each script
email = "youngil.kim@unsw.edu.au"

# ======================================================================================
# END INPUT
# Start creating the deck
# ======================================================================================


def load_config(yaml_path):
    """
    Load configuration from a YAML file.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        Config: Configuration object with loaded settings.
    """
    with open(yaml_path, "r") as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


config = load_config(yamlfile)
year = config.startyear_h
end_year = config.endyear_h

while year < end_year + 1:
    for itr in range(len(Input_var)):
        # open the sample deck
        fin = open(indeck, "r")

        # open the deck I am creating
        fout = open("runInterp_%s_%s.deck" % (Input_var[itr], year), "w")

        # Loop over the lines of the input file
        for lines in fin.readlines():

            # Replace template fields by values
            lines = lines.replace("%OUTdir%", OUT_dir)
            lines = lines.replace("%yamlfile%", yamlfile)
            lines = lines.replace("%storage%", storage)
            lines = lines.replace("%input_var%", Input_var[itr])
            lines = lines.replace("%project%", project)
            lines = lines.replace("%project2%", project2)
            lines = lines.replace("%outproject%", outproject)
            lines = lines.replace("%ncores%", str(ncores))
            lines = lines.replace("%memory%", str(memory))
            lines = lines.replace("%walltime%", str(walltime))
            lines = lines.replace("%email%", email)
            lines = lines.replace("%syear%", str(year))
            lines = lines.replace("%eyear%", str(year + 1))
            fout.write(lines)

        # Close input and output files
        fin.close()
        fout.close()

    year += 1
