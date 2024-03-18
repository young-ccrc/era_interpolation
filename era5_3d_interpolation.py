"""
This script performs horizontal and vertical interpolation of atmospheric variables from ERA5 reanalysis data to a target Global Climate Model (GCM) dataset. The interpolated data is saved as NetCDF files.

The script takes input variables from the ERA5 reanalysis data, such as specific humidity (q), temperature (t), zonal wind (u), and meridional wind (v), and maps them to the target variables in the GCM dataset, which are specific humidity (hus), air temperature (ta), zonal wind (ua), and meridional wind (va).

The horizontal interpolation is performed using xESMF library, which uses conservative or bilinear methods depending on the input variable. The vertical interpolation is performed by converting pressure levels to hybrid height coordinates using the geopotential height from ERA5 and the coefficients from the GCM dataset.

The script loops through each year and month specified in the input parameters and performs the interpolation for each variable. The interpolated data is then saved as NetCDF files.

The script uses Dask for parallel processing and xarray for data manipulation. The dask.distributed.Client is used to create a local cluster for parallel processing.

The script is designed to be run on the NCI's Gadi supercomputer, but can be modified to run on other systems.

Note: This script assumes that the necessary input files and directories are available and properly formatted.

Written by: Youngil (Young) Kim, CLEX, CCRC, UNSW
Contact: youngil.kim@unsw.edu.au
"""

import os
import warnings
from pathlib import Path

# import dask.array as da  # type: ignore
import xarray as xr  # type: ignore
import xesmf as xe  # type: ignore
from dask.distributed import Client  # type: ignore
from scipy.interpolate import interp1d

warnings.simplefilter("ignore", UserWarning)
c = Client()

# Input start --------------------------------------------------------------
input_path = (
    "/g/data/rt52/era5/pressure-levels/reanalysis"  # Path to the ERA5 reanalysis data
)
target_path = "/g/data/fs38/publications/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r1i1p1f1"  # Path to the target GCM dataset
output_path = "/scratch/dm6/yk8692/interpolation"  # Path to save the output files

input_variables = ["q", "t", "u", "v"]  # 3D variables of ERA5
target_variables = ["hus", "ta", "ua", "va"]  # 3D variables of GCM

# Target gcm information --------------------------------------------------
infor = "6hrLev"
gname = "ACCESS-ESM1-5"  # EC-Earth3-Veg
period = "historical"
cinfor = "r1i1p1f1"
sinfor = "gn"  # "gr"
version = "v20191115"

start_year = 1980  # Start year for interpolation
end_year = 1980  # End year for interpolation

# Input end ---------------------------------------------------------------

# Start of the script -----------------------------------------------------

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)
# Prepare target files mapping (horozontal interpolation)
target_files = {
    var: f"{target_path}/{infor}/{var}/{sinfor}/{version}/{var}_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{start_year}01010600-{int(start_year)+1}01010000.nc"
    for var in target_variables
}
# Load the target grid outside the loop
target_grids = {
    var: xr.open_dataset(target_files[var], chunks={"time": 500})
    for var in target_variables
}


# Define functions for the horizontal and vertical interpolation ----------------
def calculate_geopotential_height(g):
    """
    Calculate geopotential height from the geopotential variable.

    Parameters:
    g (float): The geopotential variable from ERA5, in m^2 s^-2.

    Returns:
    float: The geopotential height Z, in meters.

    Formula:
    The geopotential height Z is calculated as Z = g / g0, where g0 = 9.80665 m/s^2.
    """
    g0 = 9.80665
    Z = g / g0
    return Z


def calculate_hybrid_height(a, b, orog):
    """
    Calculates the hybrid height using coefficients from the GCM dataset and the surface orography.

    Parameters:
    a (float): Coefficient 'a' from the GCM dataset.
    b (float): Coefficient 'b' from the GCM dataset.
    orog (float): Surface orography.

    Returns:
    float: The calculated hybrid height.

    """
    z = a + b * orog
    return z


def interpolate_profile(source_profile, source_levels, target_levels):
    """
    Interpolates a source profile to match target levels.

    Args:
        source_profile (array-like): The source profile to be interpolated.
        source_levels (array-like): The levels corresponding to the source profile.
        target_levels (array-like): The target levels to interpolate the source profile to.

    Returns:
        array-like: The interpolated profile matching the target levels.
    """
    f_interp = interp1d(source_levels, source_profile, bounds_error=False, fill_value="extrapolate")  # type: ignore
    return f_interp(target_levels)


# Assuming ds_regridded, Z_era5, and gcm_z_data are xarray DataArrays/Datasets with Dask arrays
def vertical_interpolation(source_da, source_levels_da, target_levels):
    """
    Perform vertical interpolation of a source data array to target levels using xarray's apply_ufunc.

    Parameters:
        source_da (xarray.DataArray): The source data array to be interpolated.
        source_levels_da (xarray.DataArray): The source data array's levels.
        target_levels (array-like): The target levels to interpolate to.

    Returns:
        xarray.DataArray: The interpolated data array.

    """
    # Wrapper to apply interpolation using xarray's apply_ufunc to handle Dask arrays efficiently
    interpolated_da = xr.apply_ufunc(
        interpolate_profile,
        source_da,
        source_levels_da,
        target_levels,
        vectorize=True,  # Enable vectorized execution
        input_core_dims=[["level"], ["level"], ["lev"]],  # Define core dimensions
        output_core_dims=[["lev"]],  # Define output dimensions
        dask="parallelized",  # Enable Dask parallelization
        output_dtypes=[source_da.dtype],
    )

    return interpolated_da


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

# Main loop to process data for each year and month ---------------------------
for year in range(start_year, end_year + 1):
    for month in range(1, 13):  # Loop through all months
        # Vertical interpolation from pressure to hybrid height coordinate
        # This part might be modified if other GCM is used
        # Prepare target files mapping
        input_z_files = f"{input_path}/z/{start_year}/z_era5_oper_pl_{start_year}{month:02}*-{start_year}{month:02}*.nc"
        target_z_files = f"{target_path}/fx/zfull/{sinfor}/{version}/zfull_fx_{gname}_{period}_{cinfor}_{sinfor}.nc"
        target_orog_files = f"{target_path}/fx/orog/{sinfor}/{version}/orog_fx_{gname}_{period}_{cinfor}_{sinfor}.nc"

        # Load data
        era5_z_data = xr.open_mfdataset(
            input_z_files,
            combine="by_coords",
            chunks={
                "time": "auto",
                "level": "auto",
                "latitude": "auto",
                "longitude": "auto",
            },
        )
        gcm_z_data = xr.open_dataset(
            target_z_files, chunks={"time": 10, "lev": -1, "lat": -1, "lon": -1}
        )

        # resample and remap z for era5
        g_era5_resampled = era5_z_data.resample(time="6h").mean()

        for input_var, target_var in zip(input_variables, target_variables):
            input_files_pattern = (
                f"{input_path}/{input_var}/{year}/{input_var}_*_{year}{month:02d}*.nc"
            )
            output_file = (
                f"{output_path}/{target_var}_era5_oper_pl_regridded_{year}{month:02}.nc"
            )

            # Load source data
            ds = xr.open_mfdataset(
                input_files_pattern,
                combine="by_coords",
                chunks={
                    "time": "auto",
                    "level": "auto",
                    "latitude": "auto",
                    "longitude": "auto",
                },
            )
            ds_corrected = correct_latitudes(ds)
            ds_resampled = ds_corrected.resample(time="6h").mean()

            # Load the target_grid and input_var
            target_ds = target_grids[target_var]
            method = "conservative" if input_var == "q" else "bilinear"
            weights_path = f"{output_path}/weights_{target_var}_{method}.nc"
            rename_dict = {input_var: target_var}

            # Regrid
            # Original resampled data
            ds_regridded = regrid(
                ds_resampled, target_ds, method, weights_path, rename_dict
            )
            # Rechunk
            ds_regridded = ds_regridded.chunk(
                {"time": 10, "level": -1, "lat": -1, "lon": -1}
            )
            print("Horizontal interpolation complete")

            ## Vertical interpolation input ----------------------------
            # Geopotential height
            weights_path_z = f"{output_path}/weights_zfull_era5_oper_pl_{method}.nc"
            rename_dict_z = {"z": "zfull"}
            if target_var == "va":
                weights_path_va = (
                    f"{output_path}/weights_zfull_va_era5_oper_pl_{method}.nc"
                )
                g_era5_regridded = regrid(
                    g_era5_resampled, target_ds, method, weights_path_va, rename_dict_z
                )
            else:
                g_era5_regridded = regrid(
                    g_era5_resampled, target_ds, method, weights_path_z, rename_dict_z
                )

            # Perform calculations
            g_era5 = g_era5_regridded[
                "zfull"
            ]  # Assuming 'zfull' is geopotential height
            Z_era5 = calculate_geopotential_height(g_era5)

            # Rechunk
            Z_era5 = Z_era5.chunk({"time": 10, "level": -1, "lat": -1, "lon": -1})

            print("Horizontal interpolation for z complete")
            ## Vertical interpolation input end -------------------------

            # Vertical interpolation
            if target_var in ["ua", "va"]:
                weights_path_wind = (
                    f"{output_path}/weights_{target_var}_gcm_wind_{method}.nc"
                )
                gcm_z_data_interp = regrid(
                    gcm_z_data, target_ds, method, weights_path_wind, {"zfull": "zfull"}
                )
                gcm_z_data_interp = gcm_z_data_interp.chunk(
                    {"lev": -1, "lat": -1, "lon": -1}
                )
                interpolated_ds = vertical_interpolation(
                    ds_regridded[target_var], Z_era5, gcm_z_data_interp["zfull"]
                )
            else:
                interpolated_ds = vertical_interpolation(
                    ds_regridded[target_var], Z_era5, gcm_z_data["zfull"]
                )

            print("vertical interpolation complete")

            interpolated_era5_da = interpolated_ds.rename(
                {"__xarray_dataarray_variable__": target_var}
            )
            # interpolated_era5_da = xr.DataArray(
            #     interpolated_ds,
            #     dims=["time", "lev", "lat", "lon"],
            #     coords={
            #         "time": g_era5.time,
            #         "lev": gcm_z_data.lev,
            #         "lat": g_era5.latitude,
            #         "lon": g_era5.longitude,
            #     },
            # )

            original_attrs = ds_regridded.attrs.copy()
            original_global_attrs = ds_regridded.attrs.copy()

            interpolated_era5_da.attrs = original_attrs
            interpolated_era5_da.attrs["interpolation_method"] = (
                "Hybrid height cooridnates"
            )

            # Save to netcdf
            interpolated_era5_da.compute().to_netcdf(output_file)
            print(
                f"Saved regridded data for {target_var} {year}-{month:02} to {output_file}"
            )

# Close target datasets
for ds in target_grids.values():
    ds.close()

print("Interpolation complete.")
