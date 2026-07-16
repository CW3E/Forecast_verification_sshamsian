# Objective: To create a Python script that plots AR using observation data over several days from janaury 31st to February 3rd and then plots a bar graph displaying data from the cts text files
# Date: 2026-07-15
# Author: Sina Shamsian 
import pandas as pd # for data handling and working with dataframes
import numpy as np # for data handling and numerical operations
import xarray as xr # for data handling and working with netCDF files
import matplotlib.pyplot as plt # for plotting 
import glob # for file pattern matching
import re
import os # for file path handling
from matplotlib.lines import Line2D # for creating custom legend handles
import cartopy.crs as ccrs # For geographic projection and map features 
import cartopy.feature as cfeature # for adding geographic features to the map
import warnings # For filtering out warnings 
warnings.filterwarnings('ignore') # For filtering out warnings


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

def text_file_to_dataframe(file_path):
    """Read a single MET/MODE _cts.txt file into a cleaned DataFrame."""
    df = pd.read_csv(file_path, sep='\s+', header=0)  # <-- fixed: header=0, not None

    # Clean up column names
    df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True).str.lower()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    # Convert statistic columns to numeric (remove leading >= or <= if present)
    stat_cols = [
        "total","fy_oy","fy_on","fn_oy","fn_on","baser","fmean","acc","fbias",
        "pody","podn","pofd","far","csi","gss","hk","hss","odds","lodds","orss",
        "eds","seds","edi","sedi","bagss"
    ]
    for c in stat_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[<>]=?", "", regex=True), errors="coerce")

    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, subset=[col for col in ("total", "acc") if col in df.columns])

    df['source_file'] = file_path
    return df


def parse_lead_from_filename(path):
    """Extract lead hours from a mode_WestWRF_<L>0000L_... filename."""
    match = re.search(r'_(\d+)L_', os.path.basename(path))
    if match:
        return int(match.group(1)) // 10000
    return None


def load_cts_for_date(init_str, ivt_level=500,
                       base_path='/data/projects/WWRF-NRT/30YEAR-REFORECAST/MODE_verification/Raw_output/1998'):
    """Load and concatenate all _cts.txt files for one init-time directory,
    keeping only the first row per lead time."""
    pattern = f'{base_path}/{init_str}/{ivt_level}/*_cts.txt'
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'No _cts.txt files found for {init_str} at {pattern}')
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = text_file_to_dataframe(f)
        df['lead_hours'] = parse_lead_from_filename(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('lead_hours').reset_index(drop=True)

    # --- keep only the first row per lead time ---
    combined = combined.drop_duplicates(subset='lead_hours', keep='first').reset_index(drop=True)

    return combined

def clean_secondary_csv(path):
    """Read and clean a single MODE secondary-output CSV (all leads in one file)."""
    df = pd.read_csv(path, sep='\s+', engine='python', header=0)

    # Parse valid_time into a real datetime
    df['valid_time'] = pd.to_datetime(df['valid_time'], format='%Y%m%d_%H%M%S')

    # Convert lead_time from MET's HHMMSS-style int into plain hours
    df['lead_hours'] = (df['lead_time'] // 10000).astype(int)
    df = df.drop(columns=['lead_time'])

    # Round numeric columns for readability
    numeric_cols = ['centroid_lat', 'centroid_lon', 'intersect_area',
                     'fcst_90', 'obs_90', 'fcst_area', 'obs_area',
                     'fcst_angle', 'obs_angle']
    df[numeric_cols] = df[numeric_cols].round(2)

    # Reorder and sort
    cols_order = ['valid_time', 'lead_hours', 'centroid_lat', 'centroid_lon',
                  'intersect_area', 'fcst_90', 'obs_90', 'fcst_area', 'obs_area',
                  'fcst_angle', 'obs_angle']
    df = df[cols_order]
    df = df.sort_values('lead_hours').reset_index(drop=True)
    df = df.sort_values('obs_area', ascending=False).drop_duplicates(subset='lead_hours', keep='first')
    df['source_file'] = path
    return df

def load_secondary_for_date(init_str, ivt_level=500,
                              base_path='/data/projects/WWRF-NRT/30YEAR-REFORECAST/MODE_verification/Secondary_output/1998'):
    """Load and clean the secondary-output CSV for one init-time directory."""
    pattern = f'{base_path}/{init_str}/{ivt_level}/MODE_WestWRF_*.csv'
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'No secondary CSV found for {init_str} at {pattern}')
        return pd.DataFrame()

    if len(files) > 1:
        print(f'WARNING: expected 1 CSV for {init_str}, found {len(files)}: {files}')

    return clean_secondary_csv(files[0])

def load_secondary_for_dates(dates, ivt_level=500,
                               base_path='/data/projects/WWRF-NRT/30YEAR-REFORECAST/MODE_verification/Secondary_output/1998'):
    """Load and concatenate secondary-output CSVs across multiple init-time directories.

    Parameters
    ----------
    dates : list of (init_str, label) tuples
        e.g. [('1998013100', 'Jan 31, 1998 00Z'), ('1998020100', 'Feb 1, 1998 00Z'), ...]

    Returns
    -------
    dict
        Keyed by label, each value the cleaned DataFrame for that date.
    """
    result = {}
    for init_str, label in dates:
        df = load_secondary_for_date(init_str, ivt_level=ivt_level, base_path=base_path)
        if df.empty:
            print(f'No data for {label} - skipping')
            continue
        df['date_label'] = label
        result[label] = df
    return result


def dist_perf(x_moe, y_moe, normalize=True):
    """Compute the distance from the perfect forecast point (1,1) in MoE space.

    Parameters
    ----------
    x_moe, y_moe : float or array-like
        MoE coordinates (intersect/obs and intersect/fcst ratios), each in [0, 1].
    normalize : bool
        If True, divide by sqrt(2) (max possible distance) so output is in [0, 1],
        where 0 = perfect forecast and 1 = complete miss.

    Returns
    -------
    float or array-like
        Distance from perfect forecast, in [0, sqrt(2)] or [0, 1] if normalized.
    """
    dist = np.sqrt((x_moe - 1)**2 + (y_moe - 1)**2)
    if normalize:
        dist = dist / np.sqrt(2)
    return dist

def compute_moe(df):
    """Compute the Measure of Effectiveness (MoE) per DeHaan et al. 2021 (WAF).

    x_moe = intersect_area / obs_area   (false-negative measure)
    y_moe = intersect_area / fcst_area  (false-positive measure)

    Points on the y=x diagonal indicate forecast and observed objects
    are the same size (not necessarily same location).
    Above the diagonal: forecast object smaller than observed.
    Below the diagonal: forecast object larger than observed.
    """
    df = df.copy()
    df['x_moe'] = df['intersect_area'] / df['obs_area']
    df['y_moe'] = df['intersect_area'] / df['fcst_area']
    return df
# ---------------------------------------------------------------------------------# 
# Actual code starts here, after the helper functions are defined. The following section defines the dates to be sampled and the path to the data files. It also initializes an empty list to store the file paths for each date.
# --- one init-time directory per day you want to sample ---


dates = [
    ('1998013100', 'Jan 31, 1998 00Z'),
    ('1998020100', 'Feb 1, 1998 00Z'),
    ('1998020200', 'Feb 2, 1998 00Z'),
    ('1998020300', 'Feb 3, 1998 00Z'),
]
# ivt level to use for the plot, stored as part of the directory when searching for files
ivt_level = 500
base_path1 = '/data/projects/WWRF-NRT/30YEAR-REFORECAST/MODE_verification/Raw_output/1998'

day_files = []
for init_str, label in dates:
    matches = sorted(glob.glob(f'{base_path1}/{init_str}/{ivt_level}/*.nc'))
    if not matches:
        print(f'WARNING: no files found for {init_str} ({label}) - check path/date')
        continue
    # obs is identical across all lead files within one init directory,
    day_files.append((init_str, label, matches[0]))

print('Found day files:', [(d, l) for d, l, _ in day_files])

if not day_files:
    print('No files found for any of the requested dates. Check base_path/dates.') # backup error message if no files are found 
else:
    # # --- diagnostic: confirm each day's obs object and coordinates ---
    # for init_str, label, path in day_files:
    #     ds = xr.open_dataset(path)
    #     has_obs = 'obs_simp_bdy_lon' in ds
    #     n_objs = len(np.asarray(ds['obs_simp_bdy_start'])) if has_obs else 0
    #     print(f'{label}: has obs boundary var = {has_obs}, n objects = {n_objs}')
    #     if has_obs:
    #         for xs, ys in iter_objects(ds, 'obs'):
    #             print(f'   lon range {xs.min():.2f}-{xs.max():.2f}, '
    #                   f'lat range {ys.min():.2f}-{ys.max():.2f}, first point ({xs[0]:.3f}, {ys[0]:.3f})')
    #     ds.close()
    # # --- end diagnostic ---
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
#---------------------------------------------------------------------------------------#
# KEEP THIS SECTION COMMENTED OUT FOR NOW, AS IT IS NOT NEEDED FOR THE PLOT 
    # #Plots a background observed IVT field for the selected date, using a reversed plasma colormap and setting the colorbar limits to 250-1000 kg m^-1 s^-1.
    # ds_ref = xr.open_dataset(day_files[selected_index][2])
    # ds_ref['obs_raw'].plot(
    #     ax=ax, x='lon', y='lat', cmap='plasma_r', zorder=0,
    #     transform=ccrs.PlateCarree(),
    #     cbar_kwargs={'label': 'Observed IVT (kg m$^{-1}$ s$^{-1}$)'},
    #     vmin=250, vmax=1000
    # )
#----------------------------------------------------------------------------------------#
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
    ax.set_title(f'Observed AR shape evolution across multiple days')

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
# --- test the bar plotting code for HSS by valid date, grouped by lead time ------

ivt_level = 500
stat_to_plot = 'hss'  # adjust as needed
lead_hours = [24, 48, 72, 96, 120, 144, 168]
# --- load cts data for each date ---
cts_by_date = {}
for init_str, label in dates:
    df = load_cts_for_date(init_str, ivt_level=ivt_level)
    if df.empty:
        print(f'No data for {label} - skipping')
        continue
    cts_by_date[label] = df

date_labels = list(cts_by_date.keys())      # NEW line
x = np.arange(len(date_labels))             # was: np.arange(len(lead_hours))
n_leads = len(lead_hours)                   # was: n_dates = len(cts_by_date)
width = 0.8 / n_leads                       # was: 0.8 / n_dates

for label, df in cts_by_date.items():
    counts = df['lead_hours'].value_counts()
    dupes = counts[counts > 1]
    if not dupes.empty:
        print(f'{label}: duplicate lead_hours found:\n{dupes}\n')

fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.viridis(np.linspace(0.0, 0.9, n_leads))   # creates a choice of viridis colors for the bars, one for each lead time

for i, (lead, color) in enumerate(zip(lead_hours, colors)):    # was: for i, (label, df) in enumerate(cts_by_date.items()):
    values = []                                                  # NEW
    for label in date_labels:                                    # NEW inner loop
        df_indexed = cts_by_date[label].set_index('lead_hours')  # was: df.set_index('lead_hours').reindex(lead_hours)
        val = df_indexed[stat_to_plot].get(lead, np.nan)          # was: values = df_indexed[stat_to_plot].values
        values.append(val)                                       # NEW

    offset = (i - (n_leads - 1) / 2) * width    # was: (i - (n_dates - 1) / 2) * width
    ax.bar(x + offset, values, width, label=f'{lead}h', color=color)  # was: label=label (no color=)

ax.set_xticks(x)
ax.set_xticklabels(date_labels)                 # was: [f'{h}h' for h in lead_hours]
ax.set_xlabel('Valid date')                     # was: 'Forecast lead time'
ax.set_ylabel(stat_to_plot.upper())
ax.set_title(f'{stat_to_plot.upper()} by valid date, grouped by lead time')
ax.legend(title='Forecast lead', bbox_to_anchor=(1.02, 1), loc='upper left')  # was: title='Valid date', no bbox
ax.set_ylim(-1, 1)
ax.axhline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.show()

#==============================================================================#
#----- Distance from perfect forecast and MoE code grouped by lead time ------#

# --- load secondary CSVs for all dates ---
secondary_by_date = load_secondary_for_dates(dates, ivt_level=ivt_level)

# --- appling my existing compute_moe() and dist_perf() functions ---
for label, df in secondary_by_date.items():
    df = compute_moe(df)  # adds x_moe, y_moe columns
    df['dist_perf'] = dist_perf(df['x_moe'], df['y_moe'])
    secondary_by_date[label] = df

# quick check
for label, df in secondary_by_date.items():
    print(f'{label}:')
    print(df[['lead_hours', 'x_moe', 'y_moe', 'dist_perf']])

stat_to_plot = 'dist_perf'
lead_hours = [24, 48, 72, 96, 120, 144, 168]

date_labels = list(secondary_by_date.keys()) # The .keys method returns a set-like object providing a view on the dict's keys
x = np.arange(len(date_labels))
n_leads = len(lead_hours)
width = 0.8 / n_leads

fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.coolwarm(np.linspace(0.0, 0.9, n_leads))

for i, (lead, color) in enumerate(zip(lead_hours, colors)): # I used enumerate here because enumerate is useful for obtaining an indexed list for example zip(lead_hours, colors) pairs up each lead time with its matching color, e.g. (24, color0), (48, color1), (72, color2), etc ..
    values = []
    for label in date_labels:
        df_indexed = secondary_by_date[label].set_index('lead_hours')
        val = df_indexed[stat_to_plot].get(lead, np.nan)
        values.append(val)

    offset = (i - (n_leads - 1) / 2) * width
    ax.bar(x + offset, values, width, label=f'{lead}h', color=color)

ax.set_xticks(x)
ax.set_xticklabels(date_labels, rotation=30, ha='right')
ax.set_xlabel('Valid date')
ax.set_ylabel('Distance from perfect forecast')
ax.set_title('Distance from perfect forecast by valid date, grouped by lead time')
ax.legend(title='Forecast lead', bbox_to_anchor=(1.02, 1), loc='upper left')
ax.set_ylim(0, np.sqrt(2))  # normalized since your dist_perf() default is normalize=True

plt.tight_layout()
plt.show()
#=================================================================================#