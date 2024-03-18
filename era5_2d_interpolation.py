"""
This script performs horizontal interpolation of two dimensional atmospheric variables from ERA5 reanalysis data to a target Global Climate Model (GCM) dataset. The interpolated data is saved as NetCDF files.

The script takes input variables from the ERA5 reanalysis data, sea surface temperature (tos), and maps it to the target variable in the GCM dataset, which is sea surface temperature (sst).

The horizontal interpolation is performed using xESMF library, which uses conservative or bilinear methods depending on the input variable.

The script loops through each year and month specified in the input parameters and performs the interpolation for each variable. The interpolated data is then saved as NetCDF files.

The script uses Dask for parallel processing and xarray for data manipulation. The dask.distributed.Client is used to create a local cluster for parallel processing.

The script is designed to be run on the NCI's Gadi supercomputer, but can be modified to run on other systems.

Note: This script assumes that the necessary input files and directories are available and properly formatted.

Written by: Youngil (Young) Kim, CLEX, CCRC, UNSW
Contact: youngil.kim@unsw.edu.au
"""

import glob
import os

# Initialize a Dask client, optimally configured for your environment
import warnings
from pathlib import Path

import xarray as xr
import xesmf as xe  # type: ignore
from dask.distributed import Client

warnings.simplefilter("ignore", UserWarning)

c = Client()

# Input start --------------------------------------------------------------
input_path = "/g/data/rt52/era5/pressure-levels/reanalysis"
target_path = (
    "/g/data/fs38/publications/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r1i1p1f1"
)
output_path = "/scratch/dm6/yk8692/interpolation"

input_variables = ["sst"]
target_variables = ["tos"]

# target gcm informatin --------------------------------------------------
infor = "Oday"
gname = "ACCESS-ESM1-5"  # EC-Earth3-Veg
period = "historical"
cinfor = "r1i1p1f1"
sinfor = "gn"  # "gr"
version = "v20191115"

start_year = 1980
end_year = 1980
# Input end --------------------------------------------------------------

# Prepare target files mapping
target_files = {
    var: f"{target_path}/{infor}/{var}/{sinfor}/{version}/{var}_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{start_year}0101-{int(start_year)+9}1231.nc"
    for var in target_variables
}

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)


# Define functions for the horizontal and vertical interpolation ----------------
def correct_latitudes(ds):
    # Ensure latitudes are within bounds
    ds["latitude"] = ds["latitude"].clip(-90, 90)
    return ds


# Horizontal interpolation
def regrid(source_ds, target_ds, method, weights_path, rename_dict):
    """
    Perform interpolation/regridding of a source dataset to a target dataset using the specified method.

    Args:
        source_ds (xarray.Dataset): The source dataset to be regridded.
        target_ds (xarray.Dataset): The target dataset with the desired grid.
        method (str): The interpolation method to be used. Supported methods: 'bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch', 'nearest_s2d', 'nearest_d2s', 'patch'.
        weights_path (str): The path to the weights file used for regridding. If the file exists, the weights will be reused. If not, the weights will be computed and saved to this path.
        rename_dict (dict): A dictionary mapping variable names in the source dataset to the desired variable names in the regridded dataset.

    Returns:
        xarray.Dataset: The regridded dataset with variables renamed according to the provided rename_dict.

    """
    weights_file = Path(weights_path)
    reuse_weights = weights_file.is_file()
    regridder = xe.Regridder(
        source_ds,
        target_ds,
        method=method,
        filename=weights_path,
        reuse_weights=reuse_weights,
    )
    regridded_ds = regridder(source_ds).rename(rename_dict)
    return regridded_ds


# End of the functions -------------------------------------------------------


# Load the target grid outside the loop to avoid repetitive loading
target_grids = {
    var: xr.open_dataset(target_files[var], chunks={"time": 500})
    for var in target_variables
}

# Loop through each variable and perform the interpolation
for idx, target_var in enumerate(target_variables):
    target_ds = target_grids[target_var]
    weights_path = (
        f"weights_{target_var}_conservative.nc"
        if input_variables[idx] == "q"
        else f"weights_{target_var}_bilinear.nc"
    )
    rename_dict = {
        input_variables[idx]: target_var
    }  # Mapping from ERA5 to target variable names
    method = "conservative" if input_variables[idx] == "q" else "bilinear"

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):  # Adjust range as needed
            input_file = glob.glob(
                f"{input_path}/{input_variables[idx]}/{year}/{input_variables[idx]}_era5_oper_sfc_{year}{month:02}01-{year}{month:02}*.nc"
            )
            output_file = f"{output_path}/{target_variables[idx]}_era5_oper_sfc_regridded_{year}{month:02}.nc"

            # Check if any files were found
            if not input_file:  # If list is empty, no files were found
                print(f"Skipping missing files for {year}-{month:02}")
                continue

            with xr.open_mfdataset(
                input_file,
                chunks={"time": "auto", "latitude": "auto", "longitude": "auto"},
                combine="by_coords",
            ) as ds:
                ds_corrected = correct_latitudes(ds)
                print(f"Adjusted {target_var} {year}-{month:02} to 6-hourly complete")

                # Regrid the resampled dataset
                ds_regridded = regrid(
                    ds_corrected, target_ds, method, weights_path, rename_dict
                )
                ds_regridded.compute().to_netcdf(output_file)
                print(
                    f"Saved regridded data for {target_var} {year}-{month:02} to {output_file}"
                )
                # Step 1: Shift time coordinates back by 1 hour
                # ds_resampled = adjust_and_resample(ifile, ifile_next_month)

# Close target datasets
for ds in target_grids.values():
    ds.close()

print("Interpolation complete.")
