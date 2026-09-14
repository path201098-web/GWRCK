import io
import os
import zipfile
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import rasterio

from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling

from pyproj import Transformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

RASTER_FILES = {
    "ELEV": "ELEV.tif",
    "Curvatu": "Curvatu.tif",
    "Slope": "Slope.tif",
    "HillSha": "HillSha.tif",
    "TEMP": "TEMP.tif",
    "PRES": "PRES.tif",
    "EVI": "EVI.tif",
    "BSI": "BSI.tif"
}

TARGET_CRS = "EPSG:32717"
SOURCE_CRS = "EPSG:4326"
RASTER_RESOLUTION = 30.0
CN_THRESHOLD = 25.0


def calculate_vif(X, variables):

    vif_values = []

    for i in range(X.shape[1]):

        X_other = np.delete(X, i, axis=1)
        y_target = X[:, i]

        model = LinearRegression()
        model.fit(X_other, y_target)

        r2 = model.score(X_other, y_target)

        if r2 >= 0.999999:
            vif = np.inf
        else:
            vif = 1.0 / (1.0 - r2)

        vif_values.append(vif)

    return pd.DataFrame({
        "Variable": variables,
        "VIF": vif_values
    })


def bisquare_weights(distances, bandwidth):

    weights = np.zeros_like(distances, dtype=float)

    if bandwidth <= 0:
        return weights

    mask = distances < bandwidth

    weights[mask] = (
        1.0 - (distances[mask] / bandwidth) ** 2
    ) ** 2

    return weights


def calculate_local_condition_numbers(
    coords,
    X,
    bw,
    threshold=25.0
):

    n = coords.shape[0]

    distance_matrix = np.sqrt(
        (coords[:, None, 0] - coords[None, :, 0]) ** 2
        +
        (coords[:, None, 1] - coords[None, :, 1]) ** 2
    )

    local_cn = np.full(n, np.nan)

    for i in range(n):

        distances = distance_matrix[i]

        sorted_distances = np.sort(distances)

        bw_index = int(bw) - 1

        if bw_index < 0:
            bw_index = 0

        if bw_index >= len(sorted_distances):
            bw_index = len(sorted_distances) - 1

        local_bw = sorted_distances[bw_index]

        if local_bw <= 0:
            positive_distances = sorted_distances[
                sorted_distances > 0
            ]

            if len(positive_distances) == 0:
                local_bw = 1.0
            else:
                local_bw = positive_distances[0]

        weights = bisquare_weights(
            distances,
            local_bw
        )

        W = np.diag(weights)

        XtWX = X.T @ W @ X

        eigvals = np.linalg.eigvalsh(
            XtWX
        )

        eigvals = np.real(eigvals)

        eig_max = np.max(eigvals)
        eig_min = np.min(eigvals)

        if eig_min <= 0 or eig_max <= 0:
            local_cn[i] = np.inf
        else:
            local_cn[i] = np.sqrt(
                eig_max / eig_min
            )

    return local_cn, distance_matrix


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

    corrected_betas = gwr_params.copy()

    lambda_local = np.zeros(n)

    for i in range(n):

        if local_cn[i] <= CN_THRESHOLD:
            continue

        distances = distance_matrix[i]

        sorted_distances = np.sort(distances)

        bw_index = int(bw) - 1

        if bw_index < 0:
            bw_index = 0

        if bw_index >= len(sorted_distances):
            bw_index = len(sorted_distances) - 1

        local_bw = sorted_distances[bw_index]

        if local_bw <= 0:

            positive_distances = sorted_distances[
                sorted_distances > 0
            ]

            if len(positive_distances) == 0:
                local_bw = 1.0
            else:
                local_bw = positive_distances[0]

        weights = bisquare_weights(
            distances,
            local_bw
        )

        W = np.diag(weights)

        XtWX = X.T @ W @ X

        eigvals = np.linalg.eigvalsh(
            XtWX
        )

        eigvals = np.real(eigvals)

        eig_min = np.min(eigvals)

        lambda_value = 0.01 * eig_min

        if lambda_value < 0.001:
            lambda_value = 0.001

        lambda_local[i] = lambda_value

        XTWy = X.T @ W @ y

        ridge_matrix = (
            XtWX
            +
            lambda_value * np.eye(X.shape[1])
        )

        try:

            beta_ridge = np.linalg.solve(
                ridge_matrix,
                XTWy
            )

            corrected_betas[i, 1:] = (
                beta_ridge.flatten()
            )

        except np.linalg.LinAlgError:

            corrected_betas[i, 1:] = (
                gwr_params[i, 1:]
            )

    return corrected_betas, lambda_local


def run_gwr_gwrc(
    data,
    bandwidth
):

    coords = data[["X_proj", "Y_proj"]].values

    X_raw = (
        data[VARS_MODEL]
        .values
        .astype(float)
    )

    y = (
        data["SOC"]
        .values
        .astype(float)
        .reshape(-1, 1)
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(
        X_raw
    )

    y_gwr = y.flatten()

    vif_table = calculate_vif(
        X,
        VARS_MODEL
    )

    X_gwr = np.column_stack(
        [np.ones(len(X)), X]
    )

    gwr_model = GWR(
        coords,
        y_gwr,
        X,
        bw=bandwidth,
        fixed=False,
        kernel="bisquare",
        constant=True
    )

    gwr_res = gwr_model.fit()

    gwr_pred = (
        gwr_res.predy
        .flatten()
    )

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
            bandwidth,
            CN_THRESHOLD
        )
    )

    corrected_betas, lambda_local = (
        apply_gwrc(
            coords,
            X_gwr,
            y_gwr,
            gwr_res.params,
            bandwidth,
            local_cn,
            distance_matrix
        )
    )

    gwrc_pred = np.zeros(
        len(X)
    )

    for i in range(len(X)):

        x_local = np.concatenate(
            [[1.0], X[i]]
        )

        gwrc_pred[i] = (
            x_local
            @ corrected_betas[i]
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
            len(y_gwr),
            len(y_gwr)
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
            local_cn > CN_THRESHOLD
        ),

        "Lambda_Local": lambda_local
    })

    coefficient_table = pd.DataFrame(
        corrected_betas,
        columns=[
            "Intercept"
        ] + VARS_MODEL
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

        uploaded_file = uploaded_rasters[var]

        with rasterio.open(
            uploaded_file
        ) as src:

            left, bottom, right, top = (
                src.bounds
            )

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

    min_x, min_y, max_x, max_y = bounds

    width = int(
        np.ceil(
            (max_x - min_x)
            /
            RASTER_RESOLUTION
        )
    )

    height = int(
        np.ceil(
            (max_y - min_y)
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

        uploaded_file = uploaded_rasters[var]

        with rasterio.open(
            uploaded_file
        ) as src:

            destination = np.full(
                (height, width),
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

        mean = np.nanmean(arr)

        std = np.nanstd(arr)

        if std == 0 or np.isnan(std):

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
        (height, width),
        np.nan,
        dtype=np.float32
    )

    raster_gwrc = np.full(
        (height, width),
        np.nan,
        dtype=np.float32
    )

    raster_cn = np.full(
        (height, width),
        np.nan,
        dtype=np.float32
    )

    raster_lambda = np.full(
        (height, width),
        np.nan,
        dtype=np.float32
    )

    grid_x = (
        transform.c
        +
        (np.arange(width) + 0.5)
        *
        transform.a
    )

    grid_y = (
        transform.f
        +
        (np.arange(height) + 0.5)
        *
        transform.e
    )

    valid_mask = np.ones(
        (height, width),
        dtype=bool
    )

    for var in VARS_MODEL:

        valid_mask &= np.isfinite(
            raster_std[var]
        )

    valid_rows, valid_cols = (
        np.where(valid_mask)
    )

    for row, col in zip(
        valid_rows,
        valid_cols
    ):

        x_values = np.array([
            raster_std[var][row, col]
            for var in VARS_MODEL
        ])

        point_x = grid_x[col]
        point_y = grid_y[row]

        distances = np.sqrt(
            (coords[:, 0] - point_x) ** 2
            +
            (coords[:, 1] - point_y) ** 2
        )

        order = np.argsort(
            distances
        )

        k = int(
            min(
                max(1, int(bandwidth)),
                len(order)
            )
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

        local_gwr_beta = (
            weights[:, None]
            *
            gwr_params[selected]
        ).sum(axis=0)

        local_gwrc_beta = (
            weights[:, None]
            *
            corrected_betas[selected]
        ).sum(axis=0)

        x_design = np.concatenate(
            [[1.0], x_values]
        )

        raster_gwr[row, col] = (
            x_design
            @ local_gwr_beta
        )

        raster_gwrc[row, col] = (
            x_design
            @ local_gwrc_beta
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
        np.isfinite(coords[:, 0])
        &
        np.isfinite(coords[:, 1])
        &
        np.isfinite(residuals)
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

    ok = OrdinaryKriging(
        krig_x,
        krig_y,
        krig_residuals,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False,
        coordinates_type="euclidean"
    )

    increasing_y = np.sort(
        grid_y
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

    uploaded_rasters[var] = (
        st.file_uploader(
            f"{var}.tif",
            type=["tif", "tiff"],
            key=f"raster_{var}"
        )
    )


st.subheader(
    "3. Configuración del bandwidth"
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
        value=30,
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
    use_container_width=True
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
            ", ".join(missing_rasters)
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
            ", ".join(missing_columns)
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

    coords = data[
        ["X_proj", "Y_proj"]
    ].values

    X_raw = (
        data[VARS_MODEL]
        .values
        .astype(float)
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(
        X_raw
    )

    y = (
        data["SOC"]
        .values
        .astype(float)
    )

    if bw_mode == "Automático (AICc)":

        st.info(
            "Calculando bandwidth automático mediante AICc..."
        )

        selector = Sel_BW(
            coords,
            y,
            X,
            kernel="bisquare",
            fixed=False,
            constant=True
        )

        bw_opt = selector.search(
            criterion="AICc"
        )

        bandwidths_to_run = [
            int(bw_opt)
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
                result["results_table"]
                .copy()
            )

            all_results.append(
                metrics
            )

        except Exception as e:

            st.error(
                f"Error con bandwidth {bandwidth}: {e}"
            )

            st.stop()

        progress.progress(
            (i + 1)
            /
            len(bandwidths_to_run)
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
            use_container_width=True
        )

        best_row = (
            bandwidth_comparison[
                bandwidth_comparison["Model"]
                == "GWR"
            ]
            .sort_values(
                "AICc"
            )
            .iloc[0]
        )

        selected_bandwidth = int(
            best_row["Bandwidth"]
        )

        st.info(
            f"Bandwidth seleccionado para el procesamiento final: "
            f"{selected_bandwidth}"
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
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    gwr_metrics = (
        results["results_table"]
        .iloc[0]
    )

    gwrc_metrics = (
        results["results_table"]
        .iloc[1]
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
                results["local_cn"]
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
            results["gwr_res"].params,
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

    raster_cn[
        np.isfinite(raster_gwrc)
    ] = np.nan

    st.success(
        "GWRCK completado."
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
            use_container_width=True
        )

        st.download_button(
            "Descargar residuos krigeados",
            data=residual_bytes,
            file_name="Residual_GWRC_Kriged_30m.tif",
            mime="image/tiff",
            use_container_width=True
        )

        st.download_button(
            "Descargar resultados Excel",
            data=excel_bytes,
            file_name="GWRCK_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col2:

        st.download_button(
            "Descargar SOC_GWRCK_30m.tif",
            data=gwrck_bytes,
            file_name="SOC_GWRCK_30m.tif",
            mime="image/tiff",
            use_container_width=True
        )

        st.download_button(
            "Descargar todos los resultados (.zip)",
            data=zip_bytes,
            file_name="GWRCK_results.zip",
            mime="application/zip",
            use_container_width=True
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
