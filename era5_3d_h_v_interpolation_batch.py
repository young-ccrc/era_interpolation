"""
This script performs horizontal and vertical interpolation of atmospheric variables from ERA5 reanalysis data to a target Global Climate Model (GCM) dataset. The interpolated data is saved as NetCDF files.

The script takes input variables from the ERA5 reanalysis data, such as specific humidity (q), temperature (t), zonal wind (u), and meridional wind (v), and maps them to the target variables in the GCM dataset, which are specific humidity (hus), air temperature (ta), zonal wind (ua), and meridional wind (va).

The horizontal interpolation is performed using the xESMF library, which provides conservative or bilinear methods depending on the input variable. The vertical interpolation is performed by converting pressure levels to hybrid height coordinates (and hybrid sigma pressure level) using the geopotential height from ERA5 and the coefficients from the GCM dataset.

The script utilizes Dask for parallel processing and xarray for data manipulation. The dask.distributed.Client is used to create a local cluster for parallel processing.

The script is designed to be run on the NCI's Gadi supercomputer, but can be modified to run on other systems.

Note: This script assumes that the necessary input files and directories are available and properly formatted.

Author: Youngil (Young) Kim, CLEX, CCRC, UNSW
Contact: youngil.kim@unsw.edu.au
"""

import argparse
import glob
import logging
import os
import sys
import warnings

# from multiprocessing import Pool
from pathlib import Path

import dask.array as da  # type: ignore
import numpy as np  # type: ignore

# import dask.array as da  # type: ignore
import xarray as xr  # type: ignore
import xesmf as xe  # type: ignore
import yaml  # type: ignore
from dask.diagnostics import ProgressBar  # type: ignore
from dask.distributed import Client, LocalCluster  # type: ignore
from scipy.interpolate import interp1d  # type: ignore

warnings.simplefilter("ignore", UserWarning)


# Start of the script -----------------------------------------------------
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


def setup_client(n_workers=None, threads_per_worker=None):
    """
    Set up Dask client for parallel processing, allowing customization of workers and threads.

    Args:
        n_workers (int): Number of workers to use.
        threads_per_worker (int): Number of threads per worker.

    Returns:
        Client: A Dask distributed client instance.
    """
    if n_workers is None or threads_per_worker is None:
        c = Client()
    else:
        c = Client(n_workers=n_workers, threads_per_worker=threads_per_worker)
    print("Dask client setup complete.")
    return c


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run atmospheric data interpolation.")
    parser.add_argument(
        "--yp",
        type=str,
        default="./user_input_test.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--var",
        type=str,
        default=["hus", "ta", "ua", "va"],
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=28,
        help="Number of cores to use.",
    )

    parser.add_argument(
        "--mp",
        type=int,
        default=64,
        help="Amount of RAM (in GB) to allocate.",
    )

    parser.add_argument(
        "--np",
        type=int,
        default=4,
        help="Number of multiprocessing processes to use.",
    )
    parser.add_argument(
        "--mpp",
        type=int,
        default=32,
        help="Amount of RAM (in GB) to allocate per process.",
    )
    parser.add_argument("--sy", type=int, default=1982, help="Start year.")
    parser.add_argument("--ey", type=int, default=2012, help="End year.")

    return parser.parse_args()


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def robust_open_dataset(file_pattern):
    """
    Safely open a dataset with error handling.

    Args:
        file_pattern (str): Glob pattern for files to load.

    Returns:
        xarray.Dataset: The opened dataset.
    """
    try:
        dataset = xr.open_mfdataset(file_pattern, combine="by_coords")
    except FileNotFoundError:
        logging.error(f"No files found with the pattern: {file_pattern}")
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    # logging.info(f"Dataset loaded successfully from {file_pattern}")
    return dataset


# Define functions for the horizontal and vertical interpolation ----------------
def geopotential_to_geopotential_height(g):
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


def calculate_pressure_levels(ap, b, ps):
    """
    Calculate pressure at each model level by automatically broadcasting 'ap' and 'b'
    against 'ps' across their shared dimensions ('lat', 'lon') and aligning them with
    'ps' time dimension.

    Parameters:
    - ap: Xarray DataArray of 'ap' coefficient, shaped [lev]
    - b: Xarray DataArray of 'b' coefficient, shaped [lev]
    - ps: Xarray DataArray of surface pressure, shaped [time, lat, lon]

    Returns:
    - p_levels: Xarray DataArray of pressure at each model level, shaped [time, lev, lat, lon]
    """

    # Ensure 'ap' and 'b' are broadcasted and aligned along 'ps' dimensions
    # This uses Xarray's automatic alignment and broadcasting
    ap_expanded = ap * xr.ones_like(
        ps
    )  # This automatically broadcasts 'ap' across 'ps' dimensions
    b_expanded = b * xr.ones_like(ps)  # Similarly for 'b'

    # Calculate pressure at each model level
    p_levels = ap_expanded + b_expanded * ps
    return p_levels


def calculate_geopotential_height(p_levels, T_levels):
    Rd = 287.05  # J/kg/K, specific gas constant for dry air
    g = 9.80665  # m/s^2, acceleration due to gravity

    # Compute the pressure ratio without aligning by levels
    upper_p = p_levels.isel(lev=slice(None, -1)).data
    lower_p = p_levels.isel(lev=slice(1, None)).data
    p_ratio = upper_p / lower_p
    log_p_ratio = da.log(p_ratio)

    # Re-create the DataArray for log_p_ratio with adjusted coordinates
    log_p_ratio_da = xr.DataArray(
        log_p_ratio,
        dims=["time", "lev", "lat", "lon"],
        coords={
            "time": p_levels.time,
            "lev": p_levels.lev[:-1],  # Use coordinates from the upper slice
            "lat": p_levels.lat,
            "lon": p_levels.lon,
        },
    )

    # Compute mean temperature between consecutive levels without alignment
    upper_T = T_levels.isel(lev=slice(None, -1)).data
    lower_T = T_levels.isel(lev=slice(1, None)).data
    mean_T = (upper_T + lower_T) / 2

    # Re-create the DataArray for mean_T with adjusted coordinates
    mean_T_da = xr.DataArray(
        mean_T,
        dims=["time", "lev", "lat", "lon"],
        coords=log_p_ratio_da.coords,  # Match coordinates with log_p_ratio_da
    )

    # Calculate thickness of each layer (delta Z)
    delta_Z = (Rd / g) * mean_T_da * log_p_ratio_da

    # Integrate delta_Z from the top to obtain geopotential heights
    Z_levels_cumsum = delta_Z.cumsum(dim="lev")

    # Add an extra level at the top with zero or extrapolated geopotential height
    top_level = da.zeros(
        (
            Z_levels_cumsum.shape[0],
            1,
            Z_levels_cumsum.shape[2],
            Z_levels_cumsum.shape[3],
        ),
        dtype=Z_levels_cumsum.dtype,
    )
    Z_levels = da.concatenate([top_level, Z_levels_cumsum], axis=1)

    # Update level coordinates to include the top level
    Z_levels = xr.DataArray(
        Z_levels,
        dims=["time", "lev", "lat", "lon"],
        coords={
            "time": p_levels.time,
            "lev": p_levels.lev,  # Use original levels, assuming the extra level is added at the top
            "lat": p_levels.lat,
            "lon": p_levels.lon,
        },
        name="zfull",
    )

    return Z_levels


def adjust_geopotential_heights(Z_interior, original_levs):
    """
    Adjust geopotential heights by extrapolating values for the topmost and bottommost levels,
    ensuring the output aligns with the original model levels.

    Parameters:
    - Z_interior: Calculated geopotential heights for the interior levels (excluding the first and last).
    - original_levs: The original 'lev' coordinate values from the model.

    Returns:
    - Z_full: Adjusted geopotential heights including the extrapolated boundary levels.
    """
    # Extrapolate the topmost and bottommost levels
    gradient_bottom = Z_interior.isel(lev=1) - Z_interior.isel(lev=0)
    gradient_top = Z_interior.isel(lev=-1) - Z_interior.isel(lev=-2)

    bottom_extrapolated = Z_interior.isel(lev=0) - gradient_bottom
    top_extrapolated = Z_interior.isel(lev=-1) + gradient_top

    # Concatenate including the extrapolated levels
    Z_full = xr.concat([bottom_extrapolated, Z_interior, top_extrapolated], dim="lev")

    # Correctly assign the original 'lev' coordinate values to the new DataArray
    Z_full = Z_full.assign_coords(lev=original_levs)

    return Z_full


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

    # return interpolated_da
    return interpolated_da.assign_coords(lev=target_levels.lev)


def correct_latitudes(ds):
    # Ensure latitudes are within bounds
    ds["latitude"] = ds["latitude"].clip(-90, 90)
    return ds


# Horizontal interpolation
def regrid(source_ds, target_ds, method, weights_path, rename_dict):
    """
    Perform interpolation/regridding of a source dataset to a target dataset using the specified method.

    Args:
        source_ds (xarray.DataArray or xarray.Dataset): The source dataset to be regridded.
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
    if isinstance(source_ds, xr.DataArray):
        # Handling for DataArray
        if source_ds.name != "zfull":
            regridded_data = regridder(source_ds)
            # Rename the DataArray if necessary
            if source_ds.name in rename_dict:
                regridded_data.name = rename_dict[source_ds.name]
            return regridded_data
        return source_ds
    elif isinstance(source_ds, xr.Dataset):
        # Handling for Dataset
        regridded_data = regridder(source_ds)
        # Check and rename variables in the Dataset if necessary
        for var in list(regridded_data.data_vars):
            if var in rename_dict:
                regridded_data = regridded_data.rename({var: rename_dict[var]})
        return regridded_data
    else:
        raise TypeError("Input must be xarray DataArray or Dataset")


def preprocess_month(ds, year, month):
    """
    Preprocess function to select data for a specific year and month.

    Parameters:
    - ds (xarray.Dataset): Dataset to preprocess.
    - year (int): Year to filter.
    - month (int): Month to filter.

    Returns:
    - xarray.Dataset: Preprocessed dataset.
    """
    # Convert `time` to a datetime index if not already done
    # ds["time"] = xr.decode_cf(ds)["time"]

    # Filter the dataset for the specified year and month
    ds_filtered = ds.sel(time=(ds["time.year"] == year) & (ds["time.month"] == month))
    return ds_filtered


def process_data_parallel(year, config):
    # client = setup_dask(config.memory_per_process, config.num_cores_per_process)
    # client = setup_dask(config.memory_per_process, config.num_cores_per_process, 1)

    lat_max = config.lat_max
    lat_min = config.lat_min
    lon_max = config.lon_max
    lon_min = config.lon_min
    input_path = config.input_path
    target_path = config.target_path
    output_path = config.output_path
    infor = config.infor
    gname = config.gname
    period = config.period
    cinfor = config.cinfor
    sinfor = config.sinfor
    version = config.version
    input_variables = config.input_variables
    target_variables = config.target_variables
    var_interp = config.var_interp

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Define patterns to match files from the current and previous year
    current_year_pattern = f"{year}*.nc"
    selected_variables = [var_interp, "ta"]

    # Glob patterns for target files, considering the necessary period from the previous year
    target_files = {
        var: sorted(
            glob.glob(
                f"{target_path}/{infor}/{var}/{sinfor}/{version}/{var}_{infor}_{gname}_{period}_{cinfor}_{sinfor}_*{current_year_pattern}"
            )
        )
        for var in selected_variables
    }

    # for month in range(1, 13):  # Loop through all months
    # Load the target grid with preprocessing
    target_grids = {}
    for var in selected_variables:
        # Load the dataset combining files by coordinates
        # ds = xr.open_mfdataset(
        #     target_files[var],
        #     combine="by_coords",
        #     chunks={
        #         "time": "auto",
        #         "lev": "auto",
        #         "lat": "auto",
        #         "lon": "auto",
        #     },
        # )
        ds = robust_open_dataset(target_files[var])

        # Select data for a specific month and year
        # Ensure time decoding is correct (if not automatically handled by xarray)
        ds = xr.decode_cf(ds)
        selected_ds = ds.sel(time=(ds["time"].dt.year == year))
        selected_ds["lat"] = selected_ds["lat"].clip(-90, 90)

        # Check if any data is selected to prevent processing empty datasets
        if selected_ds.sizes["time"] == 0:
            raise ValueError(f"No data available for {year} in variable {var}.")

        # Store the selected dataset in the dictionary
        target_grids[var] = selected_ds

    ## GCM data
    if gname == "ACCESS-ESM1-5":  # ACCESS-ESM1-5 is hybrid height coordinate
        target_z_files = f"{target_path}/fx/zfull/{sinfor}/{version}/zfull_fx_{gname}_{period}_{cinfor}_{sinfor}.nc"
        target_zfull = xr.open_dataset(
            target_z_files,
            chunks={"time": "auto", "lev": -1, "lat": "auto", "lon": "auto"},
        )

    else:
        # Use hypsometric equation to calculate geopotential height
        target_ds = target_grids["ta"]
        target_p = calculate_pressure_levels(target_ds.ap, target_ds.b, target_ds.ps)

        target_zfull = calculate_geopotential_height(
            target_p, target_ds.ta
        )  # Calculate geopotential height

        target_zfull = target_zfull.chunk({"time": 10, "lev": -1, "lat": -1, "lon": -1})
        target_zfull = target_zfull.transpose("time", "lev", "lat", "lon")

    ## Reanalysis data

    # Prepare target files mapping
    input_z_files = f"{input_path}/z/{year}/z_era5_oper_pl_{year}*.nc"

    # Load era5 geopotential data
    # era5_z_data = xr.open_mfdataset(
    #     input_z_files,
    #     combine="by_coords",
    #     chunks={
    #         "time": "auto",
    #         "level": "auto",
    #         "latitude": "auto",
    #         "longitude": "auto",
    #     },
    # )
    era5_z_data = robust_open_dataset(input_z_files)

    # resample and remap era5 geopotential data
    era5_z_data = correct_latitudes(era5_z_data)
    g_era5_resampled = era5_z_data.resample(time="6h").mean()

    # Load ended
    if var_interp == "hus":
        var_obs = "q"
    elif var_interp == "ta":
        var_obs = "t"
    elif var_interp == "ua":
        var_obs = "u"
    elif var_interp == "va":
        var_obs = "v"

    ## Loop through each variable for interpolation (q, t, u, v)
    for input_var, target_var in zip([var_obs], [var_interp]):
        input_files_pattern = (
            f"{input_path}/{input_var}/{year}/{input_var}_*_{year}*.nc"
        )
        output_file = f"{output_path}/{target_var}_era5_oper_pl_regridded_{year}.nc"

        method = "conservative" if input_var == "q" else "bilinear"

        # Load source data
        # ds = xr.open_mfdataset(
        #     input_files_pattern,
        #     combine="by_coords",
        #     chunks={
        #         "time": "auto",
        #         "level": "auto",
        #         "latitude": "auto",
        #         "longitude": "auto",
        #     },
        # )
        ds = robust_open_dataset(input_files_pattern)

        ds_corrected = correct_latitudes(ds)
        ds_resampled = ds_corrected.resample(time="6h").mean()

        # Load the target_grid and input_var
        target_ds = target_grids[target_var]
        weights_path = f"{output_path}/weights_{target_var}_{gname}_{method}.nc"
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
        # print(
        #     f"Horizontal interpolation for {input_var} {year}-{month:02} complete."
        # )
        # print("Horizontal interpolation complete")

        weights_path_z = f"{output_path}/weights_zfull_era5_oper_pl_{gname}_bilinear.nc"
        rename_dict_z = {"z": "zfull"}

        ## Vertical interpolation input ----------------------------
        # Geopotential height
        if target_var == "va":
            weights_path_va = (
                f"{output_path}/weights_zfull_va_era5_oper_pl_{gname}_bilinear.nc"
            )
            g_era5_regridded = regrid(
                g_era5_resampled,
                target_ds,
                "bilinear",
                weights_path_va,
                rename_dict_z,
            )
        else:
            g_era5_regridded = regrid(
                g_era5_resampled,
                target_ds,
                "bilinear",
                weights_path_z,
                rename_dict_z,
            )

        # Perform calculations
        g_era5 = g_era5_regridded["zfull"]  # Assuming 'zfull' is geopotential height
        Z_era5 = geopotential_to_geopotential_height(g_era5)

        # Rechunk
        Z_era5 = Z_era5.chunk({"time": 10, "level": -1, "lat": -1, "lon": -1})

        # print(
        #     f"Horizontal interpolation for z {year}-{month:02} complete."
        # )
        ## Vertical interpolation input end -------------------------
        # print("Start vertical interpolation")
        # Vertical interpolation

        if target_var in ["ua", "va"]:
            weights_path_wind = (
                f"{output_path}/weights_{target_var}_{gname}_wind_{method}.nc"
            )
            gcm_z_data_interp = regrid(
                target_zfull,
                target_ds,
                method,
                weights_path_wind,
                {"zfull": "zfull"},
            )
            gcm_z_data_interp = gcm_z_data_interp.chunk(
                {"lev": -1, "lat": -1, "lon": -1}
            )
            interpolated_ds = vertical_interpolation(
                ds_regridded[target_var], Z_era5, gcm_z_data_interp
            )
        else:
            interpolated_ds = vertical_interpolation(
                ds_regridded[target_var], Z_era5, target_zfull
            )

        # print(
        #     f"Vertical interpolation for {input_var} {year}-{month:02} complete."
        # )

        interpolated_ds = interpolated_ds.transpose("time", "lev", "lat", "lon")
        # print("interpolated_ds: ", interpolated_ds)

        interpolated_era5_da = xr.DataArray(
            interpolated_ds,
            dims=["time", "lev", "lat", "lon"],
            coords={
                "time": interpolated_ds.time,
                "lev": interpolated_ds.lev,
                "lat": interpolated_ds.lat,
                "lon": interpolated_ds.lon,
            },
            name=target_var,
        )

        original_attrs = ds_regridded.attrs.copy()

        interpolated_era5_da.attrs = original_attrs
        interpolated_era5_da.attrs["interpolation_to"] = f"{gname}"

        sliced_obs = interpolated_era5_da.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        )
        sliced_obs = sliced_obs.chunk(
            {"time": "auto", "lev": "auto", "lat": "auto", "lon": "auto"}
        )
        # print(sliced_obs)
        output_file = f"{output_path}/{target_var}_{gname}_{year}.nc"
        print("save to netcdf")
        write_job = sliced_obs.to_netcdf(output_file, compute=False)

        with ProgressBar():
            print(f"Writing to {output_file}")
            write_job.compute()

        # Append the processed data for this variable for this month
        # data_dict[target_var].append(interpolated_era5_da)

    print(f"Interpolation for {year} complete")
    # client.close()


# End of the functions -------------------------------------------------------


def main(config):

    setup_client()

    lat_max = config.lat_max
    lat_min = config.lat_min
    lon_max = config.lon_max
    lon_min = config.lon_min
    input_path = config.input_path
    target_path = config.target_path
    output_path = config.output_path
    infor = config.infor
    gname = config.gname
    period = config.period
    cinfor = config.cinfor
    sinfor = config.sinfor
    version = config.version
    var_interp = config.var_interp
    start_year = config.startyear_h
    end_year = config.endyear_h

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    for year in range(start_year, end_year + 1):

        # Define patterns to match files from the current and previous year
        current_year_pattern = f"{year}*.nc"
        selected_variables = [var_interp, "ta"]

        # Glob patterns for target files, considering the necessary period from the previous year
        target_files = {
            var: sorted(
                glob.glob(
                    f"{target_path}/{infor}/{var}/{sinfor}/{version}/{var}_{infor}_{gname}_{period}_{cinfor}_{sinfor}_*{current_year_pattern}"
                )
            )
            for var in selected_variables
        }

        for month in range(1, 13):  # Loop through all months
            # Load the target grid with preprocessing
            target_grids = {}
            for var in selected_variables:
                # Load the dataset combining files by coordinates
                # ds = robust_open_dataset(target_files[var])
                ds = xr.open_mfdataset(
                    target_files[var],
                    combine="by_coords",
                    chunks={
                        "time": "auto",
                        "lev": "auto",
                        "lat": "auto",
                        "lon": "auto",
                    },
                )
                # Select data for a specific month and year
                # Ensure time decoding is correct (if not automatically handled by xarray)
                ds = xr.decode_cf(ds)
                # selected_ds = ds.sel(time=(ds["time"].dt.year == year))
                selected_ds = ds.sel(
                    time=(ds["time"].dt.year == year) & (ds["time"].dt.month == month)
                )
                selected_ds["lat"] = selected_ds["lat"].clip(-90, 90)

                # Check if any data is selected to prevent processing empty datasets
                if selected_ds.sizes["time"] == 0:
                    raise ValueError(
                        f"No data available for {year}-{month} in variable {var}."
                    )

                # Store the selected dataset in the dictionary
                target_grids[var] = selected_ds

            ## GCM data
            if gname == "ACCESS-ESM1-5":  # ACCESS-ESM1-5 is hybrid height coordinate
                target_z_files = f"{target_path}/fx/zfull/{sinfor}/{version}/zfull_fx_{gname}_{period}_{cinfor}_{sinfor}.nc"
                target_zfull = xr.open_dataset(
                    target_z_files,
                    chunks={"time": 10, "lev": -1, "lat": -1, "lon": -1},
                )

            else:
                # Use hypsometric equation to calculate geopotential height
                target_ds = target_grids["ta"]
                target_p = calculate_pressure_levels(
                    target_ds.ap, target_ds.b, target_ds.ps
                )

                target_zfull = calculate_geopotential_height(
                    target_p, target_ds.ta
                )  # Calculate geopotential height

                target_zfull = target_zfull.chunk(
                    {"time": 10, "lev": -1, "lat": -1, "lon": -1}
                )
                target_zfull = target_zfull.transpose("time", "lev", "lat", "lon")
                target_zfull_per = target_zfull.persist()
            ## Reanalysis data

            # Prepare target files mapping
            # input_z_files = f"{input_path}/z/{year}/z_era5_oper_pl_{year}*.nc"
            input_z_files = f"{input_path}/z/{year}/z_era5_oper_pl_{year}{month:02}*-{year}{month:02}*.nc"

            # Load era5 geopotential data
            # era5_z_data = robust_open_dataset(input_z_files)
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
            # resample and remap era5 geopotential data
            era5_z_data = correct_latitudes(era5_z_data)
            g_era5_resampled = era5_z_data.resample(time="6h").mean()

            # Load ended
            if var_interp == "hus":
                var_obs = "q"
            elif var_interp == "ta":
                var_obs = "t"
            elif var_interp == "ua":
                var_obs = "u"
            elif var_interp == "va":
                var_obs = "v"

            ## Loop through each variable for interpolation (q, t, u, v)
            for input_var, target_var in zip([var_obs], [var_interp]):
                # input_files_pattern = (
                #     f"{input_path}/{input_var}/{year}/{input_var}_*_{year}*.nc"
                # )
                # output_file = f"{output_path}/{target_var}_era5_oper_pl_regridded_{year}.nc"
                input_files_pattern = f"{input_path}/{input_var}/{year}/{input_var}_*_{year}{month:02d}*.nc"
                output_file = f"{output_path}/{target_var}_era5_oper_pl_regridded_{year}{month:02}.nc"

                method = "conservative" if input_var == "q" else "bilinear"

                # ds = robust_open_dataset(input_files_pattern)
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
                weights_path = f"{output_path}/weights_{target_var}_{gname}_{method}.nc"
                rename_dict = {input_var: target_var}

                # Regrid
                # Original resampled data
                ds_regridded = regrid(
                    ds_resampled, target_ds, method, weights_path, rename_dict
                )
                # Rechunk
                sliced_ds = ds_regridded.sel(
                    lat=slice(lat_min, lat_max),
                    lon=slice(lon_min, lon_max),
                ).chunk({"level": -1})
                sliced_ds_per = sliced_ds.persist()
                # ds_regridded = ds_regridded.chunk(
                #     {"time": 10, "level": -1, "lat": -1, "lon": -1}
                # )
                # print(
                #     f"Horizontal interpolation for {input_var} {year}-{month:02} complete."
                # )
                # print("Horizontal interpolation complete")

                weights_path_z = (
                    f"{output_path}/weights_zfull_era5_oper_pl_{gname}_bilinear.nc"
                )
                rename_dict_z = {"z": "zfull"}

                ## Vertical interpolation input ----------------------------
                # Geopotential height
                if target_var == "va":
                    weights_path_va = f"{output_path}/weights_zfull_va_era5_oper_pl_{gname}_bilinear.nc"
                    g_era5_regridded = regrid(
                        g_era5_resampled,
                        target_ds,
                        "bilinear",
                        weights_path_va,
                        rename_dict_z,
                    )
                else:
                    g_era5_regridded = regrid(
                        g_era5_resampled,
                        target_ds,
                        "bilinear",
                        weights_path_z,
                        rename_dict_z,
                    )

                sliced_g_era5_regridded = g_era5_regridded.sel(
                    lat=slice(lat_min, lat_max),
                    lon=slice(lon_min, lon_max),
                )
                # Perform calculations
                g_era5 = sliced_g_era5_regridded[
                    "zfull"
                ]  # Assuming 'zfull' is geopotential height
                Z_era5 = geopotential_to_geopotential_height(g_era5)

                # Rechunk
                # Z_era5 = Z_era5.chunk({"time": 10, "level": -1, "lat": -1, "lon": -1})
                Z_era5_per = Z_era5.persist()
                Z_era5_per = Z_era5_per.chunk({"level": -1})
                # print(
                #     f"Horizontal interpolation for z {year}-{month:02} complete."
                # )
                ## Vertical interpolation input end -------------------------
                # print("Start vertical interpolation")
                # Vertical interpolation
                if target_var in ["ua", "va"]:
                    weights_path_wind = (
                        f"{output_path}/weights_{target_var}_{gname}_wind_{method}.nc"
                    )
                    gcm_z_data_interp = regrid(
                        target_zfull_per,
                        target_ds,
                        method,
                        weights_path_wind,
                        {"zfull": "zfull"},
                    )
                    sliced_gcm_z_data_interp = gcm_z_data_interp.sel(
                        lat=slice(lat_min, lat_max),
                        lon=slice(lon_min, lon_max),
                    ).chunk({"lev": -1})
                    # gcm_z_data_interp = gcm_z_data_interp.chunk(
                    #     {"lev": -1, "lat": -1, "lon": -1}
                    # )
                    interpolated_ds = vertical_interpolation(
                        sliced_ds_per[target_var], Z_era5_per, sliced_gcm_z_data_interp
                    )
                else:
                    sliced_target_zfull = target_zfull_per.sel(
                        lat=slice(lat_min, lat_max),
                        lon=slice(lon_min, lon_max),
                    ).chunk({"lev": -1})

                    interpolated_ds = vertical_interpolation(
                        sliced_ds_per[target_var], Z_era5_per, sliced_target_zfull
                    )

                sliced_interpolated_ds_per = interpolated_ds.persist()

                # print(
                #     f"Vertical interpolation for {input_var} {year}-{month:02} complete."
                # )
                sliced_interpolated_ds_per = sliced_interpolated_ds_per.transpose(
                    "time", "lev", "lat", "lon"
                )
                # interpolated_ds = interpolated_ds.transpose("time", "lev", "lat", "lon")
                # print("interpolated_ds: ", interpolated_ds)

                interpolated_era5_da = xr.DataArray(
                    sliced_interpolated_ds_per,
                    dims=["time", "lev", "lat", "lon"],
                    coords={
                        "time": interpolated_ds.time,
                        "lev": interpolated_ds.lev,
                        "lat": interpolated_ds.lat,
                        "lon": interpolated_ds.lon,
                    },
                    name=target_var,
                )

                original_attrs = ds_regridded.attrs.copy()

                interpolated_era5_da.attrs = original_attrs
                interpolated_era5_da.attrs["interpolation_to"] = f"{gname}"

                # sliced_obs = interpolated_era5_da.sel(
                #     lat=slice(lat_min, lat_max),
                #     lon=slice(lon_min, lon_max),
                # )
                # sliced_obs = sliced_obs.chunk(
                #     {"time": "auto", "lev": "auto", "lat": "auto", "lon": "auto"}
                # )
                # print(sliced_obs)
                # output_file = f"{output_path}/{target_var}_{gname}_{year}.nc"
                output_file = f"{output_path}/{target_var}_{gname}_{year}-{month:02}.nc"
                print("save to netcdf")
                write_job = interpolated_era5_da.to_netcdf(output_file, compute=False)

                with ProgressBar():
                    print(f"Writing to {output_file}")
                    write_job.compute()

                # Append the processed data for this variable for this month
                # data_dict[target_var].append(interpolated_era5_da)

            print(f"Interpolation for year: {year} month: {month:02} complete")


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.yp)
    config.var_interp = args.var
    config.num_cores = args.nc
    config.memory = args.mp
    config.startyear_h = args.sy
    config.endyear_h = args.ey
    # config.num_processes = args.np
    # config.memory_per_process = args.mpp
    # config.num_cores_per_process = args.nc // args.np  # Distribute cores evenly
    main(config)
    print("All done!")
