import os
import zipfile
import tempfile
from pathlib import Path

import shapefile
from rasterio.mask import mask
from rasterio.warp import transform_geom

import numpy as np
import pandas as pd
import streamlit as st
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform

import folium
from folium.raster_layers import ImageOverlay
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

from pyproj import Transformer

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

from pykrige.ok import OrdinaryKriging


st.set_page_config(
    page_title="GWRCK Spatial Model",
    page_icon="🌎",
    layout="wide"
)


VARS_MODEL = [
    "ELEV",
    "Curvatu",
    "Slope",
    "HillSha",
    "TEMP",
    "PRES",
    "EVI",
    "BSI"
]

TARGET_CRS = "EPSG:32717"
SOURCE_CRS = "EPSG:4326"

RASTER_RESOLUTION = 30.0
CN_THRESHOLD = 25.0


def calculate_vif(X, variables):

    vif_values = []

    for i in range(X.shape[1]):

        X_other = np.delete(
            X,
            i,
            axis=1
        )

        y_target = X[:, i]

        model = LinearRegression()

        model.fit(
            X_other,
            y_target
        )

        r2 = model.score(
            X_other,
            y_target
        )

        if r2 >= 0.999999:
            vif = np.inf
        else:
            vif = 1.0 / (1.0 - r2)

        vif_values.append(vif)

    return pd.DataFrame({
        "Variable": variables,
        "VIF": vif_values
    })


def bisquare_weights(
    distances,
    bandwidth
):

    distances = np.asarray(
        distances,
        dtype=float
    )

    weights = np.zeros_like(
        distances,
        dtype=float
    )

    if bandwidth <= 0:
        return weights

    mask = distances < bandwidth

    weights[mask] = (
        1.0
        -
        (
            distances[mask] / bandwidth
        ) ** 2
    ) ** 2

    return weights


def get_local_bandwidth(
    distances,
    bandwidth
):

    sorted_distances = np.sort(
        np.asarray(
            distances,
            dtype=float
        )
    )

    bw_index = int(
        bandwidth
    ) - 1

    bw_index = max(
        0,
        bw_index
    )

    bw_index = min(
        bw_index,
        len(sorted_distances) - 1
    )

    local_bw = sorted_distances[
        bw_index
    ]

    if local_bw <= 0:

        positive_distances = (
            sorted_distances[
                sorted_distances > 0
            ]
        )

        if len(
            positive_distances
        ) == 0:

            local_bw = 1.0

        else:

            local_bw = positive_distances[0]

    return local_bw


def calculate_local_condition_numbers(
    coords,
    X,
    bw
):

    n = coords.shape[0]

    distance_matrix = np.sqrt(
        (
            coords[:, None, 0]
            -
            coords[None, :, 0]
        ) ** 2
        +
        (
            coords[:, None, 1]
            -
            coords[None, :, 1]
        ) ** 2
    )

    local_cn = np.full(
        n,
        np.nan,
        dtype=float
    )

    for i in range(n):

        distances = (
            distance_matrix[i]
        )

        local_bw = get_local_bandwidth(
            distances,
            bw
        )

        weights = bisquare_weights(
            distances,
            local_bw
        )

        W = np.diag(
            weights
        )

        XtWX = (
            X.T
            @ W
            @ X
        )

        eigvals = np.linalg.eigvalsh(
            XtWX
        )

        eigvals = np.real(
            eigvals
        )

        eig_max = np.max(
            eigvals
        )

        eig_min = np.min(
            eigvals
        )

        if (
            eig_min <= 0
            or
            eig_max <= 0
        ):

            local_cn[i] = np.inf

        else:

            local_cn[i] = np.sqrt(
                eig_max / eig_min
            )

    return (
        local_cn,
        distance_matrix
    )


def apply_gwrc(
    coords,
    X,
    y,
    gwr_params,
    bw,
    local_cn,
    distance_matrix
):

    n = coords.shape[0]

    corrected_betas = np.array(
        gwr_params,
        dtype=float,
        copy=True
    )

    lambda_local = np.zeros(
        n,
        dtype=float
    )

    for i in range(n):

        if local_cn[i] <= CN_THRESHOLD:
            continue

        distances = (
            distance_matrix[i]
        )

        local_bw = get_local_bandwidth(
            distances,
            bw
        )

        weights = bisquare_weights(
            distances,
            local_bw
        )

        W = np.diag(
            weights
        )

        XtWX = (
            X.T
            @ W
            @ X
        )

        eigvals = np.linalg.eigvalsh(
            XtWX
        )

        eigvals = np.real(
            eigvals
        )

        eig_min = np.min(
            eigvals
        )

        lambda_value = (
            0.01 * eig_min
        )

        if lambda_value < 0.001:
            lambda_value = 0.001

        lambda_local[i] = (
            lambda_value
        )

        XTWy = (
            X.T
            @ W
            @ y
        )

        ridge_matrix = (
            XtWX
            +
            lambda_value
            *
            np.eye(
                X.shape[1]
            )
        )

        try:

            beta_ridge = np.linalg.solve(
                ridge_matrix,
                XTWy
            )

            corrected_betas[
                i,
                1:
            ] = beta_ridge.flatten()[1:]

        except np.linalg.LinAlgError:

            corrected_betas[
                i,
                1:
            ] = gwr_params[
                i,
                1:
            ]

    return (
        corrected_betas,
        lambda_local
    )


def run_gwr_gwrc(
    data,
    bandwidth
):

    bandwidth = int(
        bandwidth
    )

    n = len(data)

    if bandwidth < 2:
        raise ValueError(
            "El bandwidth debe ser mayor o igual a 2."
        )

    if bandwidth > n:
        bandwidth = n

    coords = data[
        [
            "X_proj",
            "Y_proj"
        ]
    ].values.astype(float)

    X_raw = (
        data[VARS_MODEL]
        .values
        .astype(float)
    )

    y_gwr = (
        data["SOC"]
        .values
        .astype(float)
    )

    y_gwr_2d = y_gwr.reshape((-1, 1))

    scaler = StandardScaler()

    X = scaler.fit_transform(
        X_raw
    )

    vif_table = calculate_vif(
        X,
        VARS_MODEL
    )

    X_gwr = np.column_stack(
        [
            np.ones(n),
            X
        ]
    )

    gwr_model = GWR(
        coords,
        y_gwr_2d,
        X,
        bw=bandwidth,
        fixed=False,
        kernel="bisquare",
        constant=True,
        n_jobs=1
    )

    gwr_res = gwr_model.fit()

    gwr_params = np.asarray(
        gwr_res.params,
        dtype=float
    )

    if gwr_params.ndim == 1:

        if gwr_params.size == (
            len(VARS_MODEL) + 1
        ):

            gwr_params = np.tile(
                gwr_params,
                (n, 1)
            )

        else:

            raise ValueError(
                "Los parámetros GWR tienen una dimensión inesperada."
            )

    if gwr_params.shape != (
        n,
        len(VARS_MODEL) + 1
    ):

        raise ValueError(
            "Dimensiones inesperadas de los parámetros GWR: "
            + str(gwr_params.shape)
        )

    gwr_pred = np.asarray(
        gwr_res.predy,
        dtype=float
    ).reshape(-1)

    gwr_residuals = (
        y_gwr
        -
        gwr_pred
    )

    gwr_r2 = r2_score(
        y_gwr,
        gwr_pred
    )

    gwr_rmse = np.sqrt(
        mean_squared_error(
            y_gwr,
            gwr_pred
        )
    )

    gwr_mae = mean_absolute_error(
        y_gwr,
        gwr_pred
    )

    local_cn, distance_matrix = (
        calculate_local_condition_numbers(
            coords,
            X_gwr,
            bandwidth
        )
    )

    corrected_betas, lambda_local = (
        apply_gwrc(
            coords,
            X_gwr,
            y_gwr,
            gwr_params,
            bandwidth,
            local_cn,
            distance_matrix
        )
    )

    gwrc_pred = np.zeros(
        n,
        dtype=float
    )

    for i in range(n):

        x_local = np.concatenate(
            [
                [1.0],
                X[i]
            ]
        )

        gwrc_pred[i] = (
            x_local
            @
            corrected_betas[i]
        )

    gwrc_residuals = (
        y_gwr
        -
        gwrc_pred
    )

    gwrc_r2 = r2_score(
        y_gwr,
        gwrc_pred
    )

    gwrc_rmse = np.sqrt(
        mean_squared_error(
            y_gwr,
            gwrc_pred
        )
    )

    gwrc_mae = mean_absolute_error(
        y_gwr,
        gwrc_pred
    )

    results_table = pd.DataFrame({

        "Model": [
            "GWR",
            "GWRC"
        ],

        "N": [
            n,
            n
        ],

        "Variables": [
            len(VARS_MODEL),
            len(VARS_MODEL)
        ],

        "Bandwidth": [
            bandwidth,
            bandwidth
        ],

        "R2": [
            gwr_r2,
            gwrc_r2
        ],

        "RMSE": [
            gwr_rmse,
            gwrc_rmse
        ],

        "MAE": [
            gwr_mae,
            gwrc_mae
        ],

        "AIC": [
            gwr_res.aic,
            np.nan
        ],

        "AICc": [
            gwr_res.aicc,
            np.nan
        ]
    })

    cn_table = pd.DataFrame({

        "X": data["X"].values,

        "Y": data["Y"].values,

        "X_proj": data["X_proj"].values,

        "Y_proj": data["Y_proj"].values,

        "Local_CN": local_cn,

        "CN_Threshold": CN_THRESHOLD,

        "Ridge_Correction": (
            local_cn
            >
            CN_THRESHOLD
        ),

        "Lambda_Local": lambda_local
    })

    coefficient_table = pd.DataFrame(
        corrected_betas,
        columns=[
            "Intercept"
        ]
        +
        VARS_MODEL
    )

    coefficient_table.insert(
        0,
        "X",
        data["X"].values
    )

    coefficient_table.insert(
        1,
        "Y",
        data["Y"].values
    )

    residual_table = pd.DataFrame({

        "X": data["X"].values,

        "Y": data["Y"].values,

        "SOC_Observed": y_gwr,

        "SOC_GWR": gwr_pred,

        "Residual_GWR": gwr_residuals,

        "SOC_GWRC": gwrc_pred,

        "Residual_GWRC": gwrc_residuals
    })

    return {

        "coords": coords,

        "X": X,

        "X_gwr": X_gwr,

        "scaler": scaler,

        "gwr_res": gwr_res,

        "gwr_params": gwr_params,

        "gwr_pred": gwr_pred,

        "gwrc_pred": gwrc_pred,

        "gwrc_residuals": gwrc_residuals,

        "corrected_betas": corrected_betas,

        "local_cn": local_cn,

        "lambda_local": lambda_local,

        "distance_matrix": distance_matrix,

        "results_table": results_table,

        "cn_table": cn_table,

        "coefficient_table": coefficient_table,

        "residual_table": residual_table,

        "vif_table": vif_table
    }


def prepare_rasters(
    uploaded_rasters
):

    raster_arrays = {}

    raster_crs = None
    bounds_list = []

    for var in VARS_MODEL:

        uploaded_file = uploaded_rasters[var]

        with rasterio.open(uploaded_file) as src:

            if src.crs is None:
                raise ValueError(
                    f"El raster {var} no tiene CRS definido."
                )

            if raster_crs is None:
                raster_crs = src.crs

            bounds_list.append(
                (
                    src.bounds.left,
                    src.bounds.bottom,
                    src.bounds.right,
                    src.bounds.top
                )
            )

    left = min(
        b[0]
        for b in bounds_list
    )

    bottom = min(
        b[1]
        for b in bounds_list
    )

    right = max(
        b[2]
        for b in bounds_list
    )

    top = max(
        b[3]
        for b in bounds_list
    )

    transformer_bounds = Transformer.from_crs(
        raster_crs,
        TARGET_CRS,
        always_xy=True
    )

    xmin, ymin = transformer_bounds.transform(
        left,
        bottom
    )

    xmax, ymax = transformer_bounds.transform(
        right,
        top
    )

    width = int(
        np.ceil(
            (xmax - xmin)
            /
            RASTER_RESOLUTION
        )
    )

    height = int(
        np.ceil(
            (ymax - ymin)
            /
            RASTER_RESOLUTION
        )
    )

    if width <= 0 or height <= 0:
        raise ValueError(
            "La extensión espacial de los rasters no es válida "
            "después de transformar a EPSG:32717."
        )

    transform = from_origin(
        xmin,
        ymax,
        RASTER_RESOLUTION,
        RASTER_RESOLUTION
    )

    for var in VARS_MODEL:

        uploaded_file = uploaded_rasters[var]

        destination = np.full(
            (
                height,
                width
            ),
            np.nan,
            dtype=np.float32
        )

        with rasterio.open(
            uploaded_file
        ) as src:

            reproject(
                source=rasterio.band(
                    src,
                    1
                ),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan
            )

        raster_arrays[var] = destination

    raster_std = {}

    for var in VARS_MODEL:

        arr = raster_arrays[var].astype(float)

        mean = np.nanmean(arr)
        std = np.nanstd(arr)

        if std == 0 or np.isnan(std):

            raster_std[var] = np.zeros_like(arr)

        else:

            raster_std[var] = (arr - mean) / std

    return (
        raster_arrays,
        raster_std,
        transform,
        width,
        height
    )


def predict_raster_gwr_gwrc(
    raster_std,
    transform,
    width,
    height,
    coords,
    gwr_params,
    corrected_betas,
    X_gwr,
    bandwidth
):

    raster_gwr = np.full(
        (
            height,
            width
        ),
        np.nan,
        dtype=np.float32
    )

    raster_gwrc = np.full(
        (
            height,
            width
        ),
        np.nan,
        dtype=np.float32
    )

    raster_cn = np.full(
        (
            height,
            width
        ),
        np.nan,
        dtype=np.float32
    )

    raster_lambda = np.full(
        (
            height,
            width
        ),
        np.nan,
        dtype=np.float32
    )

    grid_x = (
        transform.c
        +
        (
            np.arange(width)
            +
            0.5
        )
        *
        transform.a
    )

    grid_y = (
        transform.f
        +
        (
            np.arange(height)
            +
            0.5
        )
        *
        transform.e
    )

    valid_mask = np.ones(
        (
            height,
            width
        ),
        dtype=bool
    )

    for var in VARS_MODEL:

        valid_mask &= np.isfinite(
            raster_std[var]
        )

    valid_rows, valid_cols = np.where(
        valid_mask
    )

    for row, col in zip(
        valid_rows,
        valid_cols
    ):

        x_values = np.array(
            [
                raster_std[var][
                    row,
                    col
                ]
                for var in VARS_MODEL
            ],
            dtype=float
        )

        point_x = grid_x[col]
        point_y = grid_y[row]

        distances = np.sqrt(
            (
                coords[:, 0]
                -
                point_x
            ) ** 2
            +
            (
                coords[:, 1]
                -
                point_y
            ) ** 2
        )

        order = np.argsort(
            distances
        )

        k = min(
            max(
                1,
                int(bandwidth)
            ),
            len(order)
        )

        selected = order[:k]

        local_distances = (
            distances[selected]
        )

        local_bw = np.max(
            local_distances
        )

        if local_bw <= 0:

            positive_distances = (
                local_distances[
                    local_distances > 0
                ]
            )

            if len(
                positive_distances
            ) == 0:

                local_bw = 1.0

            else:

                local_bw = (
                    positive_distances[0]
                )

        weights = bisquare_weights(
            local_distances,
            local_bw
        )

        weight_sum = np.sum(
            weights
        )

        if weight_sum <= 0:

            weights = np.ones(
                len(selected)
            )

            weight_sum = np.sum(
                weights
            )

        weights = (
            weights
            /
            weight_sum
        )

        local_X = X_gwr[selected]
        weighted_X = local_X * np.sqrt(weights[:, None])
        XtWX_local = weighted_X.T @ weighted_X
        eigvals_local = np.real(np.linalg.eigvalsh(XtWX_local))
        eig_max_local = np.max(eigvals_local)
        eig_min_local = np.min(eigvals_local)

        if eig_min_local <= 0 or eig_max_local <= 0:
            local_cn_value = np.inf
        else:
            local_cn_value = np.sqrt(eig_max_local / eig_min_local)

        raster_cn[row, col] = local_cn_value

        if local_cn_value > CN_THRESHOLD and eig_min_local > 0:
            local_lambda = max(0.01 * eig_min_local, 0.001)
        else:
            local_lambda = 0.0

        raster_lambda[row, col] = local_lambda

        local_gwr_beta = (
            weights[:, None]
            *
            gwr_params[selected]
        ).sum(
            axis=0
        )

        local_gwrc_beta = (
            weights[:, None]
            *
            corrected_betas[selected]
        ).sum(
            axis=0
        )

        x_design = np.concatenate(
            [
                [1.0],
                x_values
            ]
        )

        raster_gwr[
            row,
            col
        ] = (
            x_design
            @
            local_gwr_beta
        )

        raster_gwrc[
            row,
            col
        ] = (
            x_design
            @
            local_gwrc_beta
        )

    return (
        raster_gwr,
        raster_gwrc,
        raster_cn,
        raster_lambda,
        grid_x,
        grid_y
    )


def krige_residuals(
    coords,
    residuals,
    grid_x,
    grid_y
):

    valid = (
        np.isfinite(
            coords[:, 0]
        )
        &
        np.isfinite(
            coords[:, 1]
        )
        &
        np.isfinite(
            residuals
        )
    )

    krig_x = coords[
        valid,
        0
    ]

    krig_y = coords[
        valid,
        1
    ]

    krig_residuals = residuals[
        valid
    ]

    if len(
        krig_residuals
    ) < 3:

        raise ValueError(
            "No existen suficientes residuos válidos para realizar el kriging."
        )

    if np.nanstd(krig_residuals) == 0:
        kriged = np.full(
            (len(grid_y), len(grid_x)),
            float(np.nanmean(krig_residuals)),
            dtype=float
        )
        variance = np.zeros_like(kriged, dtype=float)
        return kriged, variance

    increasing_y = np.sort(np.asarray(grid_y, dtype=float))

    try:
        ok = OrdinaryKriging(
            krig_x,
            krig_y,
            krig_residuals,
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False,
            exact_values=True,
            coordinates_type="euclidean"
        )
        kriged, variance = ok.execute(
            "grid",
            grid_x,
            increasing_y
        )
    except Exception:
        ok = OrdinaryKriging(
            krig_x,
            krig_y,
            krig_residuals,
            variogram_model="linear",
            verbose=False,
            enable_plotting=False,
            exact_values=True,
            coordinates_type="euclidean"
        )
        kriged, variance = ok.execute(
            "grid",
            grid_x,
            increasing_y
        )

    kriged = np.asarray(
        kriged,
        dtype=float
    )

    variance = np.asarray(
        variance,
        dtype=float
    )

    if grid_y[0] > grid_y[-1]:

        kriged = np.flipud(
            kriged
        )

        variance = np.flipud(
            variance
        )

    return (
        kriged,
        variance
    )


def extract_clip_geometries(uploaded_zip):
    """Extrae el shapefile del ZIP y transforma sus geometrías a TARGET_CRS."""
    if uploaded_zip is None:
        return None

    extract_dir = tempfile.mkdtemp()
    zip_path = os.path.join(extract_dir, "clip_area.zip")

    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getvalue())

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    shp_files = []
    for root, _, files in os.walk(extract_dir):
        for name in files:
            if name.lower().endswith(".shp"):
                shp_files.append(os.path.join(root, name))

    if len(shp_files) != 1:
        raise ValueError(
            "El ZIP del área de recorte debe contener exactamente un archivo .shp."
        )

    shp_path = shp_files[0]
    prj_path = os.path.splitext(shp_path)[0] + ".prj"

    if not os.path.exists(prj_path):
        raise ValueError(
            "El shapefile del área de recorte debe incluir su archivo .prj."
        )

    with open(prj_path, "r", encoding="utf-8", errors="ignore") as f:
        source_wkt = f.read()

    source_crs = rasterio.crs.CRS.from_wkt(source_wkt)
    reader = shapefile.Reader(shp_path)

    geometries = []
    for shape_record in reader.iterShapeRecords():
        geom = shape_record.shape.__geo_interface__
        if geom and geom.get("coordinates"):
            geometries.append(
                transform_geom(
                    source_crs,
                    TARGET_CRS,
                    geom,
                    precision=6
                )
            )

    if not geometries:
        raise ValueError("El shapefile del área de recorte no contiene geometrías válidas.")

    return geometries


def clip_output_rasters(output_paths, clip_zip):
    """Recorta únicamente los GeoTIFF ya generados. No modifica ningún cálculo del modelo."""
    if clip_zip is None:
        return output_paths

    geometries = extract_clip_geometries(clip_zip)

    clipped_paths = dict(output_paths)

    for key, path in output_paths.items():
        if not str(path).lower().endswith((".tif", ".tiff")):
            continue

        temp_path = path + ".clipped.tif"

        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"El raster {os.path.basename(path)} no tiene CRS definido.")

            if src.crs != rasterio.crs.CRS.from_user_input(TARGET_CRS):
                raster_geometries = [
                    transform_geom(
                        TARGET_CRS,
                        src.crs,
                        geom,
                        precision=6
                    )
                    for geom in geometries
                ]
            else:
                raster_geometries = geometries

            clipped, clipped_transform = mask(
                src,
                raster_geometries,
                crop=True,
                filled=True,
                nodata=np.nan
            )

            profile = src.profile.copy()
            profile.update(
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=clipped_transform,
                dtype="float32",
                count=src.count,
                nodata=np.nan,
                compress="deflate"
            )

            with rasterio.open(temp_path, "w", **profile) as dst:
                dst.write(clipped.astype(np.float32))

        os.replace(temp_path, path)
        clipped_paths[key] = path

    return clipped_paths



def _continuous_rgba(values, vmin, vmax):
    """Convierte un raster continuo en RGBA sin modificar el GeoTIFF original."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    rgba = np.zeros(arr.shape + (4,), dtype=np.uint8)

    if not np.any(finite):
        return rgba

    if vmax <= vmin:
        vmax = vmin + 1.0

    t = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)

    # Paleta tipo viridis, adecuada para variables continuas.
    stops = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    colors = np.array([
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37]
    ], dtype=float)

    for channel in range(3):
        rgba[..., channel] = np.interp(
            t,
            stops,
            colors[:, channel]
        ).astype(np.uint8)

    rgba[..., 3] = np.where(finite, 205, 0).astype(np.uint8)
    return rgba


def _read_raster_for_webmap(path):
    """Lee un GeoTIFF y lo reproyecta solo para visualización web."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata

        if src_nodata is not None:
            data[data == src_nodata] = np.nan

        finite = np.isfinite(data)
        if not np.any(finite):
            raise ValueError(
                f"El raster {os.path.basename(path)} no contiene valores válidos para visualizar."
            )

        # Limita el tamaño de la capa web sin modificar el archivo descargable.
        max_dim = 1400
        scale = max(data.shape) / max_dim
        if scale > 1:
            dst_width = max(1, int(data.shape[1] / scale))
            dst_height = max(1, int(data.shape[0] / scale))
        else:
            dst_width = data.shape[1]
            dst_height = data.shape[0]

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            "EPSG:4326",
            src.width,
            src.height,
            *src.bounds,
            dst_width=dst_width,
            dst_height=dst_height
        )

        destination = np.full(
            (dst_height, dst_width),
            np.nan,
            dtype=np.float32
        )

        reproject(
            source=data.astype(np.float32),
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear
        )

    finite_dst = np.isfinite(destination)
    values = destination[finite_dst]
    vmin = float(np.nanpercentile(values, 2))
    vmax = float(np.nanpercentile(values, 98))

    rgba = _continuous_rgba(
        destination,
        vmin,
        vmax
    )

    left = dst_transform.c
    top = dst_transform.f
    right = left + dst_transform.a * dst_width
    bottom = top + dst_transform.e * dst_height

    bounds = [
        [bottom, left],
        [top, right]
    ]

    return rgba, bounds, vmin, vmax


def display_geotiff_map(path, title):
    """Visualización interactiva independiente del procesamiento del modelo."""
    rgba, bounds, vmin, vmax = _read_raster_for_webmap(path)

    center_lat = (bounds[0][0] + bounds[1][0]) / 2.0
    center_lon = (bounds[0][1] + bounds[1][1]) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        control_scale=True,
        tiles=None
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Imagen satelital",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True
    ).add_to(m)

    ImageOverlay(
        image=rgba,
        bounds=bounds,
        opacity=0.78,
        interactive=False,
        cross_origin=False,
        zindex=2,
        name=title
    ).add_to(m)

    colormap = LinearColormap(
        colors=[
            "#440154",
            "#3b528b",
            "#21918c",
            "#5ec962",
            "#fde725"
        ],
        vmin=vmin,
        vmax=vmax,
        caption=title
    )
    colormap.add_to(m)

    halo_css = """
    <style>
        .legend text, .legend label, .legend div {
            paint-order: stroke fill;
            stroke: white;
            stroke-width: 3px;
            stroke-linejoin: round;
            text-shadow: 0 0 3px white, 0 0 3px white;
            font-weight: 600;
        }
    </style>
    """
    m.get_root().html.add_child(folium.Element(halo_css))

    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(
        m,
        width=None,
        height=650,
        returned_objects=[]
    )


def create_geotiff(
    array,
    transform,
    output_path
):

    height, width = (
        array.shape
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate"
    }

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            array.astype(
                np.float32
            ),
            1
        )


def create_results_excel(
    results,
    output_path
):

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl"
    ) as writer:

        results[
            "results_table"
        ].to_excel(
            writer,
            sheet_name="Model_Metrics",
            index=False
        )

        results[
            "vif_table"
        ].to_excel(
            writer,
            sheet_name="VIF",
            index=False
        )

        results[
            "cn_table"
        ].to_excel(
            writer,
            sheet_name="Local_CN",
            index=False
        )

        results[
            "coefficient_table"
        ].to_excel(
            writer,
            sheet_name="Coefficients",
            index=False
        )

        results[
            "residual_table"
        ].to_excel(
            writer,
            sheet_name="Residuals",
            index=False
        )

        if "final_point_table" in results:
            results[
                "final_point_table"
            ].to_excel(
                writer,
                sheet_name="GWRCK_Final_Points",
                index=False
            )


def create_zip(
    files,
    zip_path
):

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        for file_path in files:

            z.write(
                file_path,
                arcname=os.path.basename(
                    file_path
                )
            )


st.title(
    "GWRCK Spatial Model"
)

st.write(
    "GWR → GWRC → Kriging de residuos → GWRCK"
)

st.divider()


st.subheader(
    "1. Datos de puntos"
)

excel_file = st.file_uploader(
    "Sube el archivo Excel con X, Y, SOC y las variables predictoras",
    type=["xlsx"]
)


st.subheader(
    "2. Rasters predictivos"
)

st.write(
    "Sube los ocho rasters GeoTIFF utilizados por el modelo."
)

uploaded_raster_files = st.file_uploader(
    "Selecciona los 8 rasters predictivos en bloque",
    type=["tif", "tiff"],
    accept_multiple_files=True,
    key="predictor_rasters_batch",
    help=(
        "Selecciona simultáneamente ELEV.tif, Curvatu.tif, Slope.tif, "
        "HillSha.tif, TEMP.tif, PRES.tif, EVI.tif y BSI.tif. "
        "El nombre del archivo debe identificar la variable."
    )
)

uploaded_rasters = {}
if uploaded_raster_files:
    for uploaded_file in uploaded_raster_files:
        stem = Path(uploaded_file.name).stem.strip().lower()
        for var in VARS_MODEL:
            var_norm = var.lower()
            if stem == var_norm or stem.startswith(var_norm + "_") or stem.startswith(var_norm + "-"):
                uploaded_rasters[var] = uploaded_file
                break

    missing_rasters = [
        var for var in VARS_MODEL
        if var not in uploaded_rasters
    ]

    if missing_rasters:
        st.warning(
            "Faltan los siguientes predictores: "
            + ", ".join(missing_rasters)
            + ". Selecciona los ocho archivos con sus nombres de variable correspondientes."
        )

for var in VARS_MODEL:
    uploaded_rasters.setdefault(var, None)

st.subheader(
    "3. Área de recorte (opcional)"
)

clip_zip = st.file_uploader(
    "ZIP del shapefile del área de recorte",
    type=["zip"],
    key="clip_shapefile_zip",
    help=(
        "El ZIP debe contener un único shapefile con .shp, .shx, .dbf y .prj. "
        "Si no se carga, no se aplica ningún recorte."
    )
)

st.subheader(
    "4. Configuración del bandwidth"
)

bw_mode = st.radio(
    "Método para definir bandwidth",
    [
        "Automático (AICc)",
        "Valor manual",
        "Rango"
    ]
)


manual_bw = None

if bw_mode == "Valor manual":

    manual_bw = st.number_input(
        "Bandwidth",
        min_value=2,
        value=57,
        step=1
    )


range_min = None
range_max = None
range_step = None

if bw_mode == "Rango":

    col1, col2, col3 = st.columns(3)

    with col1:

        range_min = st.number_input(
            "Mínimo",
            min_value=2,
            value=20,
            step=1
        )

    with col2:

        range_max = st.number_input(
            "Máximo",
            min_value=2,
            value=60,
            step=1
        )

    with col3:

        range_step = st.number_input(
            "Paso",
            min_value=1,
            value=5,
            step=1
        )


run_model = st.button(
    "Ejecutar GWRCK",
    type="primary",
    width="stretch"
)


if run_model:

    if excel_file is None:

        st.error(
            "Debes subir el archivo Excel."
        )

        st.stop()

    missing_rasters = [
        var
        for var in VARS_MODEL
        if uploaded_rasters.get(var) is None
    ]

    if missing_rasters:

        st.error(
            "Faltan los siguientes rasters: "
            +
            ", ".join(
                missing_rasters
            )
        )

        st.stop()

    with st.spinner(
        "Preparando datos..."
    ):

        data = pd.read_excel(
            excel_file
        )

    required_columns = [
        "X",
        "Y",
        "SOC"
    ] + VARS_MODEL

    missing_columns = [
        col
        for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:

        st.error(
            "Faltan columnas en el Excel: "
            +
            ", ".join(
                missing_columns
            )
        )

        st.stop()

    data = data[
        required_columns
    ].copy()

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna()

    if len(data) < 10:

        st.error(
            "El número de observaciones válidas es insuficiente."
        )

        st.stop()

    transformer = Transformer.from_crs(
        SOURCE_CRS,
        TARGET_CRS,
        always_xy=True
    )

    x_proj, y_proj = transformer.transform(
        data["X"].values,
        data["Y"].values
    )

    data["X_proj"] = x_proj
    data["Y_proj"] = y_proj

    n_points = len(data)

    if bw_mode == "Automático (AICc)":

        st.info(
            "Calculando bandwidth automático mediante AICc..."
        )

        X_raw_auto = (
            data[VARS_MODEL]
            .values
            .astype(float)
        )

        scaler_auto = StandardScaler()

        X_auto = scaler_auto.fit_transform(
            X_raw_auto
        )

        coords_auto = data[
            [
                "X_proj",
                "Y_proj"
            ]
        ].values.astype(float)

        y_auto = (
            data["SOC"]
            .values
            .astype(float)
        )

        selector = Sel_BW(
            coords_auto,
            y_auto.reshape((-1, 1)),
            X_auto,
            kernel="bisquare",
            fixed=False,
            constant=True,
            n_jobs=1
        )

        bw_opt = selector.search(
            criterion="AICc"
        )

        bw_array = np.asarray(
            bw_opt
        )

        if bw_array.ndim == 0:

            bw_value = bw_array.item()

        else:

            bw_value = bw_array.reshape(
                -1
            )[0]

        bandwidths_to_run = [
            int(
                round(
                    float(
                        bw_value
                    )
                )
            )
        ]

    elif bw_mode == "Valor manual":

        bandwidths_to_run = [
            int(manual_bw)
        ]

    else:

        if range_max < range_min:

            st.error(
                "El máximo debe ser mayor o igual al mínimo."
            )

            st.stop()

        bandwidths_to_run = list(
            range(
                int(range_min),
                int(range_max) + 1,
                int(range_step)
            )
        )

    bandwidths_to_run = [
        min(
            max(
                2,
                int(bw)
            ),
            n_points
        )
        for bw in bandwidths_to_run
    ]

    bandwidths_to_run = list(
        dict.fromkeys(
            bandwidths_to_run
        )
    )

    st.write(
        "Bandwidth seleccionado(s):",
        bandwidths_to_run
    )

    all_results = []

    progress = st.progress(
        0
    )

    for i, bandwidth in enumerate(
        bandwidths_to_run
    ):

        try:

            result = run_gwr_gwrc(
                data,
                bandwidth
            )

            metrics = (
                result[
                    "results_table"
                ].copy()
            )

            all_results.append(
                metrics
            )

        except Exception as e:

            import traceback

            st.error(
                f"Error con bandwidth {bandwidth}: {e}"
            )

            st.code(
                traceback.format_exc()
            )

            st.stop()

        progress.progress(
            (i + 1)
            /
            len(
                bandwidths_to_run
            )
        )

    if len(all_results) > 1:

        bandwidth_comparison = pd.concat(
            all_results,
            ignore_index=True
        )

        st.subheader(
            "Comparación de bandwidth"
        )

        st.dataframe(
            bandwidth_comparison,
            width="stretch"
        )

        best_row = (
            bandwidth_comparison[
                bandwidth_comparison[
                    "Model"
                ] == "GWR"
            ]
            .sort_values(
                "AICc"
            )
            .iloc[0]
        )

        selected_bandwidth = int(
            best_row[
                "Bandwidth"
            ]
        )

        st.info(
            "Bandwidth seleccionado para "
            "el procesamiento final: "
            +
            str(
                selected_bandwidth
            )
        )

    else:

        selected_bandwidth = (
            bandwidths_to_run[0]
        )

    with st.spinner(
        "Ejecutando modelo GWR y corrección GWRC..."
    ):

        results = run_gwr_gwrc(
            data,
            selected_bandwidth
        )

    st.success(
        "GWR y GWRC completados."
    )

    st.subheader(
        "Resultados GWR y GWRC"
    )

    visible_metrics = results["results_table"][
        ["Model", "R2", "RMSE", "MAE"]
    ].copy()
    visible_metrics.columns = [
        "Modelo", "R²", "RMSE", "MAE"
    ]

    st.dataframe(
        visible_metrics,
        width="stretch",
        hide_index=True
    )

    with st.spinner(
        "Preparando rasters..."
    ):

        (
            raster_arrays,
            raster_std,
            transform,
            width,
            height
        ) = prepare_rasters(
            uploaded_rasters
        )

    with st.spinner(
        "Calculando predicción espacial GWR y GWRC..."
    ):

        (
            raster_gwr,
            raster_gwrc,
            raster_cn,
            raster_lambda,
            grid_x,
            grid_y
        ) = predict_raster_gwr_gwrc(
            raster_std,
            transform,
            width,
            height,
            results["coords"],
            results["gwr_params"],
            results["corrected_betas"],
            results["X_gwr"],
            selected_bandwidth
        )

    with st.spinner(
        "Ejecutando kriging de residuos GWRC..."
    ):

        (
            kriged_residual,
            kriging_variance
        ) = krige_residuals(
            results["coords"],
            results["gwrc_residuals"],
            grid_x,
            grid_y
        )

    raster_gwrck = (
        raster_gwrc
        +
        kriged_residual
    )

    # Validación GWRCK en los puntos originales:
    # se extrae el residuo krigeado del raster en cada punto y se suma
    # a la predicción GWRC calculada directamente en ese punto.
    # Esto evita validar contra una segunda predicción GWRC generada
    # desde los valores estandarizados de los rasters.
    point_coords = [
        (float(x), float(y))
        for x, y in results["coords"]
    ]

    residual_sampled = np.full(
        len(data),
        np.nan,
        dtype=float
    )

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=kriged_residual.shape[0],
            width=kriged_residual.shape[1],
            count=1,
            dtype="float64",
            crs=TARGET_CRS,
            transform=transform,
            nodata=np.nan
        ) as src_residual:
            src_residual.write(
                np.asarray(kriged_residual, dtype=np.float64),
                1
            )

            sampled = list(
                rasterio.sample.sample_gen(
                    src_residual,
                    point_coords,
                    indexes=1,
                    masked=True
                )
            )

            for i, value in enumerate(sampled):
                value = np.ma.asarray(value)
                if value.size > 0 and not np.ma.is_masked(value[0]):
                    residual_sampled[i] = float(value[0])

    gwrck_extracted = (
        results["gwrc_pred"].astype(float)
        +
        residual_sampled
    )

    y_eval = data["SOC"].values.astype(float)
    valid_eval = np.isfinite(gwrck_extracted) & np.isfinite(y_eval)

    if np.sum(valid_eval) < 2:
        raise ValueError(
            "No hay suficientes valores válidos del raster GWRCK para calcular las métricas finales."
        )

    gwrck_r2 = r2_score(
        y_eval[valid_eval],
        gwrck_extracted[valid_eval]
    )
    gwrck_rmse = np.sqrt(
        mean_squared_error(
            y_eval[valid_eval],
            gwrck_extracted[valid_eval]
        )
    )
    gwrck_mae = mean_absolute_error(
        y_eval[valid_eval],
        gwrck_extracted[valid_eval]
    )

    gwrck_metrics = pd.DataFrame({
        "Model": ["GWRCK"],
        "N": [int(np.sum(valid_eval))],
        "Variables": [len(VARS_MODEL)],
        "Bandwidth": [selected_bandwidth],
        "R2": [gwrck_r2],
        "RMSE": [gwrck_rmse],
        "MAE": [gwrck_mae]
    })

    final_results_table = pd.concat(
        [results["results_table"], gwrck_metrics],
        ignore_index=True
    )
    results["results_table"] = final_results_table

    results["gwrck_extracted"] = gwrck_extracted

    final_point_table = results["residual_table"].copy()
    final_point_table["Residual_GWRC_Kriged"] = (
        gwrck_extracted - results["gwrc_pred"]
    )
    final_point_table["SOC_GWRCK_Extracted"] = gwrck_extracted
    results["final_point_table"] = final_point_table

    st.success(
        "GWRCK completado."
    )

    st.subheader(
        "Resultados finales GWRCK"
    )

    visible_final_metrics = results["results_table"][
        ["Model", "R2", "RMSE", "MAE"]
    ].copy()
    visible_final_metrics.columns = [
        "Modelo", "R²", "RMSE", "MAE"
    ]

    st.dataframe(
        visible_final_metrics,
        width="stretch",
        hide_index=True
    )

    st.subheader(
        "Estructura final del GWRCK"
    )

    st.code(
        "SOC_GWRCK = SOC_GWRC + Kriged_Residual_GWRC"
    )

    temp_dir = tempfile.mkdtemp()

    gwr_path = os.path.join(
        temp_dir,
        "SOC_GWR_30m.tif"
    )

    gwrc_path = os.path.join(
        temp_dir,
        "SOC_GWRC_30m.tif"
    )

    residual_path = os.path.join(
        temp_dir,
        "Residual_GWRC_Kriged_30m.tif"
    )

    variance_path = os.path.join(
        temp_dir,
        "Kriging_Variance_30m.tif"
    )

    gwrck_path = os.path.join(
        temp_dir,
        "SOC_GWRCK_30m.tif"
    )

    cn_path = os.path.join(
        temp_dir,
        "CN_Local_30m.tif"
    )

    lambda_path = os.path.join(
        temp_dir,
        "Lambda_Local_30m.tif"
    )

    excel_path = os.path.join(
        temp_dir,
        "GWRCK_results.xlsx"
    )

    create_geotiff(
        raster_gwr,
        transform,
        gwr_path
    )

    create_geotiff(
        raster_gwrc,
        transform,
        gwrc_path
    )

    create_geotiff(
        kriged_residual,
        transform,
        residual_path
    )

    create_geotiff(
        kriging_variance,
        transform,
        variance_path
    )

    create_geotiff(
        raster_gwrck,
        transform,
        gwrck_path
    )

    create_geotiff(
        raster_cn,
        transform,
        cn_path
    )

    create_geotiff(
        raster_lambda,
        transform,
        lambda_path
    )

    create_results_excel(
        results,
        excel_path
    )

    output_files = [
        gwr_path,
        gwrc_path,
        residual_path,
        variance_path,
        gwrck_path,
        cn_path,
        lambda_path,
        excel_path
    ]

    output_paths = {
        "gwr": gwr_path,
        "gwrc": gwrc_path,
        "residual": residual_path,
        "variance": variance_path,
        "gwrck": gwrck_path,
        "cn": cn_path,
        "lambda": lambda_path
    }

    if clip_zip is not None:
        with st.spinner("Aplicando recorte espacial al área del shapefile..."):
            output_paths = clip_output_rasters(output_paths, clip_zip)
        st.success("Recorte espacial aplicado correctamente a los GeoTIFF. Las métricas del modelo no fueron modificadas.")

    zip_path = os.path.join(
        temp_dir,
        "GWRCK_results.zip"
    )

    output_files = [
        output_paths["gwr"],
        output_paths["gwrc"],
        output_paths["residual"],
        output_paths["variance"],
        output_paths["gwrck"],
        output_paths["cn"],
        output_paths["lambda"],
        excel_path
    ]

    create_zip(
        output_files,
        zip_path
    )

    st.subheader(
        "Visualización espacial"
    )

    st.write(
        "Visualización espacial del resultado final SOCS sobre una base satelital. "
        "Las demás capas se conservan únicamente como resultados descargables."
    )

    with st.spinner("Preparando visualización espacial..."):
        display_geotiff_map(
            output_paths["gwrck"],
            "SOCS (Mg ha)"
        )

    st.subheader(
        "Descargar resultados"
    )

    st.subheader(
        "Descargar resultados individuales"
    )

    download_labels = {
        "gwr": "Descargar SOC_GWR_30m.tif",
        "gwrc": "Descargar SOC_GWRC_30m.tif",
        "residual": "Descargar Residual_GWRC_Kriged_30m.tif",
        "variance": "Descargar Kriging_Variance_30m.tif",
        "gwrck": "Descargar SOC_GWRCK_30m.tif",
        "cn": "Descargar CN_Local_30m.tif",
        "lambda": "Descargar Lambda_Local_30m.tif"
    }

    download_names = {
        "gwr": "SOC_GWR_30m.tif",
        "gwrc": "SOC_GWRC_30m.tif",
        "residual": "Residual_GWRC_Kriged_30m.tif",
        "variance": "Kriging_Variance_30m.tif",
        "gwrck": "SOC_GWRCK_30m.tif",
        "cn": "CN_Local_30m.tif",
        "lambda": "Lambda_Local_30m.tif"
    }

    download_items = list(download_labels.keys())
    for start in range(0, len(download_items), 2):
        cols = st.columns(2)
        for col, key in zip(cols, download_items[start:start + 2]):
            with open(output_paths[key], "rb") as f:
                file_bytes = f.read()
            with col:
                st.download_button(
                    download_labels[key],
                    data=file_bytes,
                    file_name=download_names[key],
                    mime="image/tiff",
                    width="stretch",
                    key=f"download_{key}"
                )

    st.download_button(
        "Descargar resultados Excel",
        data=excel_bytes,
        file_name="GWRCK_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        key="download_excel_final"
    )

    st.download_button(
        "Descargar todos los resultados (.zip)",
        data=zip_bytes,
        file_name="GWRCK_results.zip",
        mime="application/zip",
        width="stretch",
        key="download_zip_final"
    )

    st.subheader(
        "Archivos generados"
    )

    st.write(
        """
        • SOC_GWR_30m.tif
        • SOC_GWRC_30m.tif
        • Residual_GWRC_Kriged_30m.tif
        • Kriging_Variance_30m.tif
        • SOC_GWRCK_30m.tif
        • CN_Local_30m.tif
        • Lambda_Local_30m.tif
        • GWRCK_results.xlsx
        """
    )
