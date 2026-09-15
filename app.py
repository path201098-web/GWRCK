import os
import zipfile
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import rasterio

from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_geom
from rasterio.mask import mask
from rasterio.io import MemoryFile
import shapefile

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
            ] = beta_ridge.flatten()

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

    bounds = None

    for var in VARS_MODEL:

        uploaded_file = (
            uploaded_rasters[var]
        )

        with rasterio.open(
            uploaded_file
        ) as src:

            left = src.bounds.left
            bottom = src.bounds.bottom
            right = src.bounds.right
            top = src.bounds.top

            if bounds is None:

                bounds = [
                    left,
                    bottom,
                    right,
                    top
                ]

            else:

                bounds[0] = min(
                    bounds[0],
                    left
                )

                bounds[1] = min(
                    bounds[1],
                    bottom
                )

                bounds[2] = max(
                    bounds[2],
                    right
                )

                bounds[3] = max(
                    bounds[3],
                    top
                )

    min_x, min_y, max_x, max_y = (
        bounds
    )

    width = int(
        np.ceil(
            (
                max_x - min_x
            )
            /
            RASTER_RESOLUTION
        )
    )

    height = int(
        np.ceil(
            (
                max_y - min_y
            )
            /
            RASTER_RESOLUTION
        )
    )

    transform = from_origin(
        min_x,
        max_y,
        RASTER_RESOLUTION,
        RASTER_RESOLUTION
    )

    for var in VARS_MODEL:

        uploaded_file = (
            uploaded_rasters[var]
        )

        with rasterio.open(
            uploaded_file
        ) as src:

            destination = np.full(
                (
                    height,
                    width
                ),
                np.nan,
                dtype=np.float32
            )

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

            raster_arrays[var] = (
                destination
            )

    raster_std = {}

    for var in VARS_MODEL:

        arr = raster_arrays[var]

        mean = np.nanmean(
            arr
        )

        std = np.nanstd(
            arr
        )

        if (
            std == 0
            or
            np.isnan(std)
        ):

            raster_std[var] = (
                np.zeros_like(arr)
            )

        else:

            raster_std[var] = (
                arr - mean
            ) / std

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


def extract_clip_geometries(zip_file):

    temp_shape_dir = tempfile.mkdtemp()

    try:
        with zipfile.ZipFile(zip_file) as z:
            members = z.namelist()
            shapefile_members = [
                name for name in members
                if name.lower().endswith(".shp")
            ]

            if len(shapefile_members) == 0:
                raise ValueError(
                    "El ZIP del área de recorte no contiene ningún archivo .shp."
                )

            if len(shapefile_members) > 1:
                raise ValueError(
                    "El ZIP contiene más de un shapefile .shp. "
                    "Deja únicamente el shapefile del área que deseas utilizar."
                )

            z.extractall(temp_shape_dir)

        shp_name = shapefile_members[0]
        shp_path = os.path.join(
            temp_shape_dir,
            shp_name
        )

        if not os.path.exists(shp_path):
            candidates = []
            for root, _, files in os.walk(temp_shape_dir):
                for filename in files:
                    if filename.lower().endswith(".shp"):
                        candidates.append(
                            os.path.join(root, filename)
                        )

            if len(candidates) != 1:
                raise ValueError(
                    "No se pudo localizar correctamente el shapefile dentro del ZIP."
                )

            shp_path = candidates[0]

        # Lectura del shapefile mediante PyShp.
        # Esto evita depender de Fiona/GDAL adicional en Streamlit Cloud.
        try:
            reader = shapefile.Reader(shp_path)
        except Exception as e:
            raise ValueError(
                "No se pudo abrir el shapefile de recorte. "
                "Verifica que el ZIP contenga .shp, .shx y .dbf."
            ) from e

        geometries = []
        for shape_record in reader.iterShapeRecords():
            geometry = shape_record.shape.__geo_interface__
            if geometry is not None:
                geometries.append(geometry)

        reader.close()

        if len(geometries) == 0:
            raise ValueError(
                "El shapefile de recorte no contiene geometrías válidas."
            )

        # El CRS se obtiene del archivo .prj asociado al .shp.
        prj_path = os.path.splitext(shp_path)[0] + ".prj"

        if not os.path.exists(prj_path):
            # Algunos ZIP conservan diferencias de mayúsculas/minúsculas.
            prj_candidates = []
            shp_base = os.path.splitext(
                os.path.basename(shp_path)
            )[0].lower()

            for root, _, files in os.walk(temp_shape_dir):
                for filename in files:
                    if (
                        os.path.splitext(filename)[0].lower() == shp_base
                        and filename.lower().endswith(".prj")
                    ):
                        prj_candidates.append(
                            os.path.join(root, filename)
                        )

            if len(prj_candidates) == 1:
                prj_path = prj_candidates[0]

        if not os.path.exists(prj_path):
            raise ValueError(
                "El shapefile no tiene información de CRS. "
                "Incluye el archivo .prj dentro del ZIP."
            )

        with open(
            prj_path,
            "r",
            encoding="utf-8-sig"
        ) as prj_file:
            source_crs = prj_file.read().strip()

        if not source_crs:
            raise ValueError(
                "El archivo .prj está vacío. "
                "No se puede determinar el CRS del shapefile."
            )

        target_geometries = [
            transform_geom(
                source_crs,
                TARGET_CRS,
                geometry,
                precision=-1
            )
            for geometry in geometries
        ]

        return target_geometries

    finally:
        # El directorio temporal se mantiene durante el procesamiento
        # del shapefile y se elimina al finalizar la lectura.
        import shutil
        shutil.rmtree(
            temp_shape_dir,
            ignore_errors=True
        )


def clip_raster_arrays(
    raster_arrays,
    transform,
    clip_zip
):

    geometries = extract_clip_geometries(
        clip_zip
    )

    clipped = {}
    clipped_transform = None

    for name, array in raster_arrays.items():

        height, width = array.shape

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": TARGET_CRS,
            "transform": transform,
            "nodata": np.nan
        }

        with MemoryFile() as memfile:

            with memfile.open(
                **profile
            ) as dataset:

                dataset.write(
                    np.asarray(
                        array,
                        dtype=np.float32
                    ),
                    1
                )

                out_image, out_transform = mask(
                    dataset,
                    geometries,
                    crop=True,
                    filled=True,
                    nodata=np.nan
                )

        clipped[name] = out_image[0]
        clipped_transform = out_transform

    return clipped, clipped_transform


def clip_output_arrays(
    arrays,
    transform,
    clip_zip
):

    clipped, clipped_transform = clip_raster_arrays(
        arrays,
        transform,
        clip_zip
    )

    return clipped, clipped_transform


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

uploaded_rasters = {}

for var in VARS_MODEL:

    uploaded_rasters[var] = st.file_uploader(
        f"{var}.tif",
        type=["tif", "tiff"],
        key=f"raster_{var}"
    )


st.subheader(
    "3. Área de recorte (opcional)"
)

clip_zip = st.file_uploader(
    "Sube opcionalmente un ZIP con el shapefile del área de recorte",
    type=["zip"],
    key="clip_shapefile_zip",
    help=(
        "El ZIP debe contener el .shp, .shx, .dbf y .prj del área. "
        "Si no subes un ZIP, los resultados se generan para toda la extensión."
    )
)

if clip_zip is not None:
    st.info(
        "Área de recorte activada. Los GeoTIFF generados se recortarán al área del shapefile."
    )
else:
    st.caption(
        "Sin ZIP: no se aplicará ningún recorte espacial."
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
        if uploaded_rasters[var] is None
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

    st.dataframe(
        results["results_table"],
        width="stretch"
    )

    col1, col2, col3 = st.columns(3)

    gwr_metrics = (
        results[
            "results_table"
        ].iloc[0]
    )

    gwrc_metrics = (
        results[
            "results_table"
        ].iloc[1]
    )

    with col1:

        st.metric(
            "GWR R²",
            f"{gwr_metrics['R2']:.4f}"
        )

    with col2:

        st.metric(
            "GWRC R²",
            f"{gwrc_metrics['R2']:.4f}"
        )

    with col3:

        corrected_points = int(
            np.sum(
                results[
                    "local_cn"
                ]
                >
                CN_THRESHOLD
            )
        )

        st.metric(
            "Puntos corregidos",
            corrected_points
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

    point_rows, point_cols = rasterio.transform.rowcol(
        transform,
        results["coords"][:, 0],
        results["coords"][:, 1]
    )

    point_rows = np.asarray(point_rows, dtype=int)
    point_cols = np.asarray(point_cols, dtype=int)

    valid_point_extract = (
        (point_rows >= 0)
        & (point_rows < height)
        & (point_cols >= 0)
        & (point_cols < width)
    )

    gwrck_extracted = np.full(
        len(data),
        np.nan,
        dtype=float
    )

    gwrck_extracted[valid_point_extract] = raster_gwrck[
        point_rows[valid_point_extract],
        point_cols[valid_point_extract]
    ]

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

    # El recorte es únicamente espacial y se aplica después de calcular
    # GWR, GWRC y GWRCK y después de calcular sus métricas.
    # Por tanto, no modifica el modelo ni sus métricas.
    output_transform = transform

    if clip_zip is not None:

        with st.spinner(
            "Aplicando recorte espacial al área seleccionada..."
        ):

            arrays_to_clip = {
                "GWR": raster_gwr,
                "GWRC": raster_gwrc,
                "Residual_GWRC_Kriged": kriged_residual,
                "Kriging_Variance": kriging_variance,
                "GWRCK": raster_gwrck,
                "CN_Local": raster_cn,
                "Lambda_Local": raster_lambda
            }

            try:
                clipped_arrays, output_transform = clip_output_arrays(
                    arrays_to_clip,
                    transform,
                    clip_zip
                )

                raster_gwr = clipped_arrays["GWR"]
                raster_gwrc = clipped_arrays["GWRC"]
                kriged_residual = clipped_arrays["Residual_GWRC_Kriged"]
                kriging_variance = clipped_arrays["Kriging_Variance"]
                raster_gwrck = clipped_arrays["GWRCK"]
                raster_cn = clipped_arrays["CN_Local"]
                raster_lambda = clipped_arrays["Lambda_Local"]

                st.success(
                    "Recorte espacial aplicado correctamente."
                )

            except Exception as e:

                st.error(
                    "No fue posible aplicar el shapefile de recorte: "
                    + str(e)
                )
                st.stop()

    st.success(
        "GWRCK completado."
    )

    st.subheader(
        "Resultados finales GWRCK"
    )

    st.dataframe(
        results["results_table"],
        width="stretch"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("GWRCK R²", f"{gwrck_r2:.4f}")
    with col2:
        st.metric("GWRCK RMSE", f"{gwrck_rmse:.4f}")
    with col3:
        st.metric("GWRCK MAE", f"{gwrck_mae:.4f}")

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
        output_transform,
        gwr_path
    )

    create_geotiff(
        raster_gwrc,
        output_transform,
        gwrc_path
    )

    create_geotiff(
        kriged_residual,
        output_transform,
        residual_path
    )

    create_geotiff(
        kriging_variance,
        output_transform,
        variance_path
    )

    create_geotiff(
        raster_gwrck,
        output_transform,
        gwrck_path
    )

    create_geotiff(
        raster_cn,
        output_transform,
        cn_path
    )

    create_geotiff(
        raster_lambda,
        output_transform,
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

    zip_path = os.path.join(
        temp_dir,
        "GWRCK_results.zip"
    )

    create_zip(
        output_files,
        zip_path
    )

    st.subheader(
        "Descargar resultados"
    )

    col1, col2 = st.columns(2)

    with open(
        gwrc_path,
        "rb"
    ) as f:

        gwrc_bytes = f.read()

    with open(
        gwrck_path,
        "rb"
    ) as f:

        gwrck_bytes = f.read()

    with open(
        residual_path,
        "rb"
    ) as f:

        residual_bytes = f.read()

    with open(
        zip_path,
        "rb"
    ) as f:

        zip_bytes = f.read()

    with open(
        excel_path,
        "rb"
    ) as f:

        excel_bytes = f.read()

    with col1:

        st.download_button(
            "Descargar SOC_GWRC_30m.tif",
            data=gwrc_bytes,
            file_name="SOC_GWRC_30m.tif",
            mime="image/tiff",
            width="stretch"
        )

        st.download_button(
            "Descargar residuos krigeados",
            data=residual_bytes,
            file_name="Residual_GWRC_Kriged_30m.tif",
            mime="image/tiff",
            width="stretch"
        )

        st.download_button(
            "Descargar resultados Excel",
            data=excel_bytes,
            file_name="GWRCK_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch"
        )

    with col2:

        st.download_button(
            "Descargar SOC_GWRCK_30m.tif",
            data=gwrck_bytes,
            file_name="SOC_GWRCK_30m.tif",
            mime="image/tiff",
            width="stretch"
        )

        st.download_button(
            "Descargar todos los resultados (.zip)",
            data=zip_bytes,
            file_name="GWRCK_results.zip",
            mime="application/zip",
            width="stretch"
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
