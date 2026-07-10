# Objective: To create a Python script that plots AR using observation data over several days from janaury 31st to February 3rd
# Date: 2026-07-09
# Author: Sina Shamsian 
import numpy as np # for data handling and numerical operations
import xarray as xr # for data handling and working with netCDF files
import matplotlib.pyplot as plt # for plotting 
import glob # for file pattern matching
from matplotlib.lines import Line2D # for creating custom legend handles
import cartopy.crs as ccrs # For geographic projection and map features 
import cartopy.feature as cfeature # for adding geographic features to the map


# --- helper functions ----------------
# The function below finds the boundary index varibales for a simplifie MODE object variable preflix like "obs"
# This function mainly locates the obs_simp_bdy_start plus one of obs_simp_bdy_npts, obs_simp_bdy_lens, or obs_simp_bdy_count.
def _bdy_vars(ds, prefix):
    """Resolve the boundary index variable names for a MODE object dataset.
    Mode object boundaries are stored as a start index array plus one of several count-style arrays. This helper findsd:
    - `{prefix}_simp_bdy_start`
      - one of `{prefix}_simp_bdy_npts`, `{prefix}_simp_bdy_lens`, or
        `{prefix}_simp_bdy_count`

    Args:
        ds: xarray.Dataset containing MODE object fields.
        prefix: String prefix for the object type, e.g. "obs".

    Returns:
        A tuple (start_var, count_var) with the variable names.

    Raises:
        KeyError: if the expected start or count variable cannot be found.
    """
    start = f"{prefix}_simp_bdy_start"
    npts  = next((f"{prefix}_simp_bdy_{s}" for s in ("npts", "lens", "count")
                  if f"{prefix}_simp_bdy_{s}" in ds), None)
    if start not in ds or npts is None:
        raise KeyError(f"Couldn't find {prefix} boundary index vars. Available: "
                       f"{[v for v in ds.variables if 'simp_bdy' in v]}")
    return start, npts
# This function reads the longitude and lattide arrays for a given simplified MODE object and yields the closed polygon coordinates for each object.
def iter_objects(ds, prefix):
    """Yield closed polygon coordinates for each simplified MODE object.
    Reads simplified boundary lon/lat arrays and uses start/count index arrays to reconstruct 
    each object contour as a closed polygon
    Args:
        ds: xarray.Dataset containing MODE object fields.
        prefix: String prefix for the object type, e.g. "obs".

    Yields:
        Tuples (xs, ys) where xs and ys are 1D numpy arrays of longitude
        and latitude coordinates for a single object, with the first point
        repeated at the end to close the polygon.
    """

    
    lon = np.asarray(ds[f"{prefix}_simp_bdy_lon"])
    lat = np.asarray(ds[f"{prefix}_simp_bdy_lat"])
    s_name, n_name = _bdy_vars(ds, prefix)
    start = np.asarray(ds[s_name]).astype(int)
    npts  = np.asarray(ds[n_name]).astype(int)
    for s, n in zip(start, npts):
        xs, ys = lon[s:s+n], lat[s:s+n]
        yield np.append(xs, xs[0]), np.append(ys, ys[0])

# --------------------------------------------------------------------------
# --- one init-time directory per day you want to sample ---

dates = [
    ('1998013100', 'Jan 31, 1998 00Z'),
    ('1998020100', 'Feb 1, 1998 00Z'),
    ('1998020200', 'Feb 2, 1998 00Z'),
    ('1998020300', 'Feb 3, 1998 00Z'),
]

# Print the available dates for the user to select from
print('Available dates:')
for i, (init_str, label) in enumerate(dates):
    print(f'  {i}: {label}')
# --- prompt user to select one date for the title's valid time ---
selection = input('Enter the number of the date to use as the title valid time, 0 stands for the first date and 3 stands for the final date: ')
try:
    selected_index = int(selection)
    selected_label = dates[selected_index][1]
except (ValueError, IndexError):
    print('Invalid selection - defaulting to the first date in the list.')
    selected_label = dates[0][1]
# ivt level to use for the plot, stored as part of the directory when searching for files
ivt_level = 500
base_path = '/data/projects/WWRF-NRT/30YEAR-REFORECAST/MODE_verification/Raw_output/1998'

day_files = []
for init_str, label in dates:
    matches = sorted(glob.glob(f'{base_path}/{init_str}/{ivt_level}/*.nc'))
    if not matches:
        print(f'WARNING: no files found for {init_str} ({label}) - check path/date')
        continue
    # obs is identical across all lead files within one init directory,
    day_files.append((init_str, label, matches[0]))

print('Found day files:', [(d, l) for d, l, _ in day_files])

if not day_files:
    print('No files found for any of the requested dates. Check base_path/dates.') # backup error message if no files are found 
else:
    # --- diagnostic: confirm each day's obs object and coordinates ---
    for init_str, label, path in day_files:
        ds = xr.open_dataset(path)
        has_obs = 'obs_simp_bdy_lon' in ds
        n_objs = len(np.asarray(ds['obs_simp_bdy_start'])) if has_obs else 0
        print(f'{label}: has obs boundary var = {has_obs}, n objects = {n_objs}')
        if has_obs:
            for xs, ys in iter_objects(ds, 'obs'):
                print(f'   lon range {xs.min():.2f}-{xs.max():.2f}, '
                      f'lat range {ys.min():.2f}-{ys.max():.2f}, first point ({xs[0]:.3f}, {ys[0]:.3f})')
        ds.close()
    # --- end diagnostic ---
    # The following code block below builds the map for the background of the plot
    proj = ccrs.LambertConformal(
        central_longitude=-140,
        central_latitude=45,
        standard_parallels=(30, 60)
    )

    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={'projection': proj})
    ax.set_extent([-170, -110, 25, 62], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor='#cfe3ee', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#f0efe9', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            category='cultural', name='admin_1_states_provinces_lines',
            scale='50m', facecolor='none'
        ),
        edgecolor='gray', linewidth=0.6, zorder=1
    )

    #Plots a background observed IVT field for the selected date, using a reversed plasma colormap and setting the colorbar limits to 250-1000 kg m^-1 s^-1.
    ds_ref = xr.open_dataset(day_files[selected_index][2])
    ds_ref['obs_raw'].plot(
        ax=ax, x='lon', y='lat', cmap='plasma_r', zorder=0,
        transform=ccrs.PlateCarree(),
        cbar_kwargs={'label': 'Observed IVT (kg m$^{-1}$ s$^{-1}$)'},
        vmin=250, vmax=1000
    )

    # One OBSERVED outline per day, colored by day.
    manual_colors = ['#00FF00', '#00FFFF', '#0000FF', '#FF00FF']  # green, cyan, blue, magenta 
    colors = manual_colors[:len(day_files)]
    handles = []

    for (init_str, label, path), color in zip(day_files, colors):
        ds = xr.open_dataset(path)
        if 'obs_simp_bdy_lon' not in ds:
            print(f'No obs boundary found for {label}')
            continue
        for xs, ys in iter_objects(ds, 'obs'):
            ax.plot(xs, ys, color=color, lw=2.2, zorder=5,
                     transform=ccrs.PlateCarree())
        handles.append(Line2D([0], [0], color=color, lw=2.2, label=label))
        ds.close()

    ax.legend(handles=handles, title='Observed valid time', loc='upper right', framealpha=0.9)
    ax.set_title(f'Observed AR shape evolution across multiple days valid time {selected_label}')

    gl = ax.gridlines(linewidth=0.3, color='gray', alpha=0.5,crs=ccrs.PlateCarree())

    gl.top_labels = False
    gl.right_labels = False
    # --- manual lon/lat tick labels (safe fallback, avoids GEOSException) ---
    for lon in [-170, -160, -150, -140, -130, -120, -110]:
        ax.text(lon, 24, f'{abs(lon)}°W', transform=ccrs.PlateCarree(),
                 ha='center', va='top', fontsize=8)

    for lat in [25, 30, 35, 40, 45, 50, 55, 60]:
        ax.text(-171, lat, f'{lat}°N', transform=ccrs.PlateCarree(),
                 ha='right', va='center', fontsize=8)
    # --- end manual labels ---
    plt.tight_layout()
    plt.show()
