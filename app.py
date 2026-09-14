import os
import warnings
warnings.filterwarnings("ignore")

import io
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import rasterio

from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from pyproj import Transformer

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.linear_model import LinearRegression

from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

from scipy.interpolate import griddata

from pykrige.ok import OrdinaryKriging


st.set_page_config(
    page_title="GWR-GWRC-GWRCK Spatial Model",
    layout="wide"
)

st.title("GWR-GWRC-GWRCK Spatial Model")

st.write(
    "GWR → GWRC → Kriging de residuos → GWRCK"
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

CN_THRESHOLD = 25

RESOLUTION = 30.0


@st.cache_data(show_spinner=False)
def save_uploaded_file(
    uploaded_file
):

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]

    fd, path = tempfile.mkstemp(
        suffix=suffix
    )

    os.close(fd)

    with open(
        path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    return path


def bisquare_weights(
    distances,
    bandwidth
):

    distances = np.asarray(
        distances,
        dtype=float
    )

    w = np.zeros_like(
        distances,
        dtype=float
    )

    if (
        not np.isfinite(bandwidth)
        or
        bandwidth <= 0
    ):

        return w

    mask = (
        distances < bandwidth
    )

    w[mask] = (
        1
        -
        (
            distances[mask]
            /
            bandwidth
        ) ** 2
    ) ** 2

    return w


def calculate_local_cn(
    X,
    distance_matrix,
    bandwidth
):

    n = X.shape[0]

    local_CN = np.zeros(n)

    for i in range(n):

        distances = (
            distance_matrix[i]
        )

        sorted_d = np.sort(
            distances
        )

        bw_local = sorted_d[
            min(
                int(bandwidth) - 1,
                n - 1
            )
        ]

        weights = bisquare_weights(
            distances,
            bw_local
        )

        W = np.diag(
            weights
        )

        XtWX = (
            X.T
            @ W
            @ X
        )

        try:

            eigenvalues = np.linalg.eigvalsh(
                XtWX
            )

            eigenvalues = eigenvalues[
                eigenvalues > 1e-10
            ]

            if len(eigenvalues) > 0:

                cn = np.sqrt(
                    eigenvalues.max()
                    /
                    eigenvalues.min()
                )

            else:

                cn = np.nan

        except Exception:

            cn = np.nan

        local_CN[i] = cn

    return local_CN


def calculate_gwrc(
    X,
    y,
    gwr_betas,
    distance_matrix,
    bandwidth,
    local_CN
):

    n = X.shape[0]

    local_lambda = np.zeros(n)

    corrected_betas = (
        gwr_betas.copy()
    )

    for i in range(n):

        if (
            np.isfinite(
                local_CN[i]
            )
            and
            local_CN[i] > CN_THRESHOLD
        ):

            distances = (
                distance_matrix[i]
            )

            sorted_d = np.sort(
                distances
            )

            bw_local = sorted_d[
                min(
                    int(bandwidth) - 1,
                    n - 1
                )
            ]

            weights = bisquare_weights(
                distances,
                bw_local
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

            eigvals = eigvals[
                eigvals > 1e-10
            ]

            if len(eigvals) > 0:

                eig_min = eigvals.min()

                lambda_local = (
                    0.01
                    *
                    eig_min
                )

                if (
                    lambda_local <= 0
                ):

                    lambda_local = 0.001

            else:

                lambda_local = 0.001

            local_lambda[i] = (
                lambda_local
            )

            I = np.eye(
                X.shape[1]
            )

            try:

                beta_ridge = np.linalg.solve(
                    XtWX
                    +
                    lambda_local * I,
                    X.T
                    @ W
                    @ y
                )

                corrected_betas[
                    i,
                    1:
                ] = (
                    beta_ridge.flatten()
                )

            except Exception:

                corrected_betas[i] = (
                    gwr_betas[i]
                )

    gwrc_pred = np.zeros(n)

    for i in range(n):

        beta = (
            corrected_betas[i]
        )

        x_local = np.concatenate(
            [
                [1.0],
                X[i]
            ]
        )

        gwrc_pred[i] = (
            x_local
            @
            beta
        )

    return (
        corrected_betas,
        local_lambda,
        gwrc_pred
    )


def run_gwr_gwrc(
    data,
    bandwidth,
    automatic_bw=False
):

    coords = data[
        [
            "X_proj",
            "Y_proj"
        ]
    ].values.astype(float)

    X_raw = data[
        VARS_MODEL
    ].values.astype(float)

    scaler = StandardScaler()

    X = scaler.fit_transform(
        X_raw
    )

    vif_values = []

    for i, var in enumerate(
        VARS_MODEL
    ):

        y_vif = X[:, i]

        X_vif = np.delete(
            X,
            i,
            axis=1
        )

        reg = LinearRegression()

        reg.fit(
            X_vif,
            y_vif
        )

        r2 = reg.score(
            X_vif,
            y_vif
        )

        if r2 >= 0.999999:

            vif = np.inf

        else:

            vif = 1 / (
                1 - r2
            )

        vif_values.append(
            vif
        )

    vif_table = pd.DataFrame({

        "Variable": VARS_MODEL,

        "VIF": vif_values

    })

    y = data[
        "SOC"
    ].values.astype(float)

    y_gwr = y.reshape(
        (-1, 1)
    )

    if automatic_bw:

        selector = Sel_BW(
            coords,
            y_gwr,
            X,
            kernel="bisquare",
            fixed=False,
            constant=True,
            n_jobs=1
        )

        bw_opt = selector.search(
            criterion="AICc"
        )

        bw_opt = int(
            round(
                float(bw_opt)
            )
        )

    else:

        bw_opt = int(
            bandwidth
        )

    n = len(data)

    if bw_opt < 2:

        raise ValueError(
            "El bandwidth debe ser mayor o igual a 2."
        )

    if bw_opt > n:

        bw_opt = n

    gwr_model = GWR(
        coords,
        y_gwr,
        X,
        bw=bw_opt,
        fixed=False,
        kernel="bisquare",
        constant=True,
        n_jobs=1
    )

    gwr_res = gwr_model.fit()

    gwr_pred = (
        gwr_res.predy
        .flatten()
    )

    gwr_residuals = (
        y
        -
        gwr_pred
    )

    r2_gwr = r2_score(
        y,
        gwr_pred
    )

    rmse_gwr = np.sqrt(
        mean_squared_error(
            y,
            gwr_pred
        )
    )

    mae_gwr = mean_absolute_error(
        y,
        gwr_pred
    )

    gwr_betas = (
        gwr_res.params
    )

    dx = (
        coords[:, None, 0]
        -
        coords[None, :, 0]
    )

    dy = (
        coords[:, None, 1]
        -
        coords[None, :, 1]
    )

    distance_matrix = np.sqrt(
        dx**2
        +
        dy**2
    )

    local_CN = calculate_local_cn(
        X,
        distance_matrix,
        bw_opt
    )

    (
        corrected_betas,
        local_lambda,
        gwrc_pred
    ) = calculate_gwrc(
        X,
        y_gwr,
        gwr_betas,
        distance_matrix,
        bw_opt,
        local_CN
    )

    r2_gwrc = r2_score(
        y,
        gwrc_pred
    )

    rmse_gwrc = np.sqrt(
        mean_squared_error(
            y,
            gwrc_pred
        )
    )

    mae_gwrc = mean_absolute_error(
        y,
        gwrc_pred
    )

    residual_gwrc = (
        y
        -
        gwrc_pred
    )

    results = pd.DataFrame({

        "Modelo": [
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
            bw_opt,
            bw_opt
        ],

        "R2_InSample": [
            r2_gwr,
            r2_gwrc
        ],

        "RMSE_InSample": [
            rmse_gwr,
            rmse_gwrc
        ],

        "MAE_InSample": [
            mae_gwr,
            mae_gwrc
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

        "X_projected": coords[:, 0],

        "Y_projected": coords[:, 1],

        "CN_Local": local_CN,

        "CN_GT_25": (
            local_CN
            >
            CN_THRESHOLD
        ),

        "Lambda_Local":
            local_lambda

    })

    coef_columns = [
        "Intercept"
    ] + VARS_MODEL

    coef_table = pd.DataFrame(
        corrected_betas,
        columns=coef_columns
    )

    coef_table.insert(
        0,
        "Y",
        data["Y"].values
    )

    coef_table.insert(
        0,
        "X",
        data["X"].values
    )

    coef_table["SOC_GWR"] = (
        gwr_pred
    )

    coef_table["SOC_GWRC"] = (
        gwrc_pred
    )

    residual_table = pd.DataFrame({

        "X": data["X"].values,

        "Y": data["Y"].values,

        "SOC_Observed": y,

        "SOC_GWR": gwr_pred,

        "Residual_GWR":
            y - gwr_pred,

        "SOC_GWRC": gwrc_pred,

        "Residual_GWRC":
            residual_gwrc,

        "CN_Local": local_CN,

        "Lambda_Local":
            local_lambda

    })

    summary = pd.DataFrame({

        "Parametro": [

            "N",

            "Variables",

            "Bandwidth",

            "CN_threshold",

            "N_CN_gt_25",

            "Percent_CN_gt_25",

            "CN_min",

            "CN_max",

            "CN_mean",

            "Lambda_min",

            "Lambda_max",

            "Lambda_mean"

        ],

        "Valor": [

            n,

            len(VARS_MODEL),

            bw_opt,

            CN_THRESHOLD,

            int(
                np.sum(
                    local_CN
                    >
                    CN_THRESHOLD
                )
            ),

            (
                100
                *
                np.sum(
                    local_CN
                    >
                    CN_THRESHOLD
                )
                /
                n
            ),

            np.nanmin(
                local_CN
            ),

            np.nanmax(
                local_CN
            ),

            np.nanmean(
                local_CN
            ),

            local_lambda.min(),

            local_lambda.max(),

            local_lambda.mean()

        ]

    })

    return {

        "coords": coords,

        "X": X,

        "scaler": scaler,

        "gwr_res": gwr_res,

        "gwr_params":
            gwr_betas,

        "corrected_betas":
            corrected_betas,

        "gwr_pred":
            gwr_pred,

        "gwrc_pred":
            gwrc_pred,

        "gwr_residuals":
            gwr_residuals,

        "residual_gwrc":
            residual_gwrc,

        "bw_opt":
            bw_opt,

        "local_CN":
            local_CN,

        "local_lambda":
            local_lambda,

        "distance_matrix":
            distance_matrix,

        "results":
            results,

        "vif_table":
            vif_table,

        "cn_table":
            cn_table,

        "coef_table":
            coef_table,

        "residual_table":
            residual_table,

        "summary":
            summary,

        "r2_gwr":
            r2_gwr,

        "rmse_gwr":
            rmse_gwr,

        "mae_gwr":
            mae_gwr,

        "r2_gwrc":
            r2_gwrc,

        "rmse_gwrc":
            rmse_gwrc,

        "mae_gwrc":
            mae_gwrc

    }


def create_rasters(
    data,
    results,
    raster_paths
):

    coords = results[
        "coords"
    ]

    corrected_betas = results[
        "corrected_betas"
    ]

    bw_opt = results[
        "bw_opt"
    ]

    raster_crs = None

    bounds_list = []

    for path in raster_paths.values():

        with rasterio.open(path) as src:

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

    transformer_bounds = (
        Transformer.from_crs(
            raster_crs,
            TARGET_CRS,
            always_xy=True
        )
    )

    xmin, ymin = (
        transformer_bounds.transform(
            left,
            bottom
        )
    )

    xmax, ymax = (
        transformer_bounds.transform(
            right,
            top
        )
    )

    width = int(
        np.ceil(
            (
                xmax - xmin
            )
            /
            RESOLUTION
        )
    )

    height = int(
        np.ceil(
            (
                ymax - ymin
            )
            /
            RESOLUTION
        )
    )

    transform = from_origin(
        xmin,
        ymax,
        RESOLUTION,
        RESOLUTION
    )

    raster_arrays = {}

    for variable, path in (
        raster_paths.items()
    ):

        destination = np.full(
            (
                height,
                width
            ),
            np.nan,
            dtype=np.float32
        )

        with rasterio.open(
            path
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

        raster_arrays[
            variable
        ] = destination

    raster_std = {}

    for variable in VARS_MODEL:

        arr = (
            raster_arrays[
                variable
            ].astype(float)
        )

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

            raster_std[
                variable
            ] = np.zeros_like(
                arr
            )

        else:

            raster_std[
                variable
            ] = (
                arr - mean
            ) / std

    gwrc_raster = np.full(
        (
            height,
            width
        ),
        np.nan,
        dtype=np.float32
    )

    rows_idx, cols_idx = np.indices(
        (
            height,
            width
        )
    )

    grid_x = (
        transform.c
        +
        (
            cols_idx + 0.5
        )
        *
        transform.a
    )

    grid_y = (
        transform.f
        +
        (
            rows_idx + 0.5
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

    for variable in VARS_MODEL:

        valid_mask &= np.isfinite(
            raster_std[
                variable
            ]
        )

    valid_rows, valid_cols = (
        np.where(
            valid_mask
        )
    )

    progress = st.progress(
        0
    )

    total = len(
        valid_rows
    )

    for count, (r, c) in enumerate(
        zip(
            valid_rows,
            valid_cols
        )
    ):

        x_values = np.array([

            raster_std[v][r, c]

            for v in VARS_MODEL

        ])

        if not np.all(
            np.isfinite(
                x_values
            )
        ):

            continue

        gx = grid_x[r, c]

        gy = grid_y[r, c]

        distances = np.sqrt(

            (
                coords[:, 0]
                -
                gx
            ) ** 2

            +

            (
                coords[:, 1]
                -
                gy
            ) ** 2

        )

        sorted_idx = np.argsort(
            distances
        )

        idx = sorted_idx[
            :
            min(
                int(bw_opt),
                len(coords)
            )
        ]

        local_dist = distances[
            idx
        ]

        local_bw = (
            local_dist[-1]
        )

        weights = bisquare_weights(
            local_dist,
            local_bw
        )

        if (
            np.sum(weights)
            <=
            0
        ):

            continue

        w_norm = (
            weights
            /
            np.sum(weights)
        )

        beta_local_corrected = (
            corrected_betas[idx]
        )

        beta_gwrc = np.sum(

            beta_local_corrected
            *
            w_norm[:, None],

            axis=0

        )

        x_design = np.concatenate(

            [
                [1.0],
                x_values
            ]

        )

        pred_gwrc = (
            x_design
            @
            beta_gwrc
        )

        gwrc_raster[
            r,
            c
        ] = pred_gwrc

        if (
            count
            %
            max(
                1,
                total // 100
            )
            ==
            0
        ):

            progress.progress(

                min(
                    1.0,
                    count
                    /
                    max(
                        1,
                        total
                    )
                )

            )

    progress.progress(
        1.0
    )

    grid_points = np.column_stack(
        [
            coords[:, 0],
            coords[:, 1]
        ]
    )

    grid_coordinates = np.column_stack(
        [
            grid_x.ravel(),
            grid_y.ravel()
        ]
    )

    cn_grid = griddata(
        grid_points,
        results[
            "local_CN"
        ],
        grid_coordinates,
        method="nearest"
    )

    cn_grid = (
        cn_grid.reshape(
            height,
            width
        ).astype(
            np.float32
        )
    )

    lambda_grid = griddata(
        grid_points,
        results[
            "local_lambda"
        ],
        grid_coordinates,
        method="nearest"
    )

    lambda_grid = (
        lambda_grid.reshape(
            height,
            width
        ).astype(
            np.float32
        )
    )

    residuals = results[
        "residual_gwrc"
    ]

    residual_mask = (

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
        residual_mask,
        0
    ]

    krig_y = coords[
        residual_mask,
        1
    ]

    krig_z = residuals[
        residual_mask
    ]

    if len(krig_z) < 3:

        raise ValueError(
            "No hay suficientes residuos válidos "
            "para realizar el kriging."
        )

    kriging_progress = st.progress(
        0
    )

    st.write(
        "Ajustando Ordinary Kriging de los residuos GWRC..."
    )

    ok = OrdinaryKriging(

        krig_x,

        krig_y,

        krig_z,

        variogram_model="spherical",

        nlags=12,

        weight=False,

        exact_values=True,

        verbose=False

    )

    kriging_progress.progress(
        0.25
    )

    valid_grid_x = grid_x[
        valid_mask
    ]

    valid_grid_y = grid_y[
        valid_mask
    ]

    unique_x = np.unique(
        valid_grid_x
    )

    unique_y = np.unique(
        valid_grid_y
    )

    residual_kriging_raster = np.full(
        (
            height,
            width
        ),
        np.nan,
        dtype=np.float32
    )

    st.write(
        "Interpolando los residuos sobre la malla de 30 m..."
    )

    try:

        z_kriged, ss_kriged = (
            ok.execute(
                "points",
                valid_grid_x,
                valid_grid_y
            )
        )

        residual_kriging_raster[
            valid_mask
        ] = np.asarray(
            z_kriged,
            dtype=float
        ).reshape(-1)

    except Exception:

        z_kriged, ss_kriged = (
            ok.execute(
                "grid",
                unique_x,
                unique_y
            )
        )

        grid_kriged = np.asarray(
            z_kriged,
            dtype=float
        )

        for i, y_value in enumerate(
            unique_y
        ):

            row_mask = np.isclose(
                grid_y,
                y_value
            )

            residual_kriging_raster[
                row_mask
            ] = grid_kriged[i, :]

    kriging_progress.progress(
        0.75
    )

    soc_gwrck_raster = (

        gwrc_raster

        +

        residual_kriging_raster

    )

    kriging_progress.progress(
        1.0
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

        "compress": "lzw"

    }

    output_paths = {}

    raster_outputs = {

        "SOC_GWRC_30m.tif":
            gwrc_raster,

        "Residual_GWRC_Kriging_30m.tif":
            residual_kriging_raster,

        "SOC_GWRCK_30m.tif":
            soc_gwrck_raster,

        "CN_Local_30m.tif":
            cn_grid,

        "Lambda_Local_30m.tif":
            lambda_grid

    }

    for filename, array in (
        raster_outputs.items()
    ):

        path = os.path.join(

            tempfile.gettempdir(),

            filename

        )

        with rasterio.open(

            path,

            "w",

            **profile

        ) as dst:

            dst.write(

                array.astype(
                    np.float32
                ),

                1

            )

        output_paths[
            filename
        ] = path

    return (

        output_paths,

        krig_z

    )


st.sidebar.header(
    "Datos de entrada"
)

excel_upload = (
    st.sidebar.file_uploader(
        "Excel con datos de SOC",
        type=["xlsx"]
    )
)

st.sidebar.subheader(
    "Rasters predictoras"
)

raster_uploads = {}

for variable in VARS_MODEL:

    uploaded = (
        st.sidebar.file_uploader(
            f"{variable}.tif",
            type=["tif", "tiff"],
            key=f"raster_{variable}"
        )
    )

    raster_uploads[
        variable
    ] = uploaded


st.sidebar.subheader(
    "Configuración GWR"
)

automatic_bw = (
    st.sidebar.checkbox(
        "Calcular bandwidth automáticamente con AICc",
        value=False
    )
)

bandwidth = (
    st.sidebar.number_input(
        "Bandwidth",
        min_value=2,
        max_value=94,
        value=57,
        step=1,
        disabled=automatic_bw
    )
)

st.sidebar.subheader(
    "Configuración Kriging"
)

st.sidebar.write(
    "Ordinary Kriging"
)

st.sidebar.write(
    "Semivariograma: Spherical"
)

st.sidebar.write(
    "Número de lags: 12"
)

run_model = (
    st.sidebar.button(
        "Ejecutar GWR + GWRC + Kriging",
        type="primary",
        use_container_width=True
    )
)


if excel_upload is None:

    st.info(
        "Carga el archivo Excel para comenzar."
    )

    st.stop()


missing_rasters = [

    variable

    for variable, uploaded
    in raster_uploads.items()

    if uploaded is None

]


if missing_rasters:

    st.warning(

        "Faltan estos rasters: "
        +
        ", ".join(
            missing_rasters
        )

    )

    st.stop()


if not run_model:

    st.info(

        "Carga todos los rasters y pulsa "
        "'Ejecutar GWR + GWRC + Kriging'."

    )

    st.stop()


try:

    with st.spinner(
        "Preparando datos..."
    ):

        excel_path = (
            save_uploaded_file(
                excel_upload
            )
        )

        raster_paths = {}

        for variable, uploaded in (
            raster_uploads.items()
        ):

            raster_paths[
                variable
            ] = save_uploaded_file(
                uploaded
            )

        data = pd.read_excel(
            excel_path
        )

        required_columns = [

            "X",
            "Y",
            "SOC"

        ] + VARS_MODEL

        missing = [

            c

            for c in required_columns

            if c not in data.columns

        ]

        if missing:

            raise ValueError(

                "Faltan columnas en el Excel: "
                +
                str(missing)

            )

        data = data.replace(

            [
                np.inf,
                -np.inf
            ],

            np.nan

        )

        data = data.dropna(

            subset=required_columns

        ).copy()

        if len(data) < 2:

            raise ValueError(

                "No hay suficientes registros válidos."

            )

        lon = data[
            "X"
        ].astype(float).values

        lat = data[
            "Y"
        ].astype(float).values

        transformer = (
            Transformer.from_crs(

                "EPSG:4326",

                TARGET_CRS,

                always_xy=True

            )
        )

        X_proj, Y_proj = (
            transformer.transform(
                lon,
                lat
            )
        )

        data["X_proj"] = X_proj

        data["Y_proj"] = Y_proj

    st.success(

        f"Datos preparados: "
        f"{len(data)} observaciones válidas."

    )

    with st.spinner(
        "Ejecutando GWR y GWRC..."
    ):

        result = run_gwr_gwrc(

            data,

            bandwidth,

            automatic_bw

        )

    st.success(
        "GWR y GWRC ejecutados correctamente."
    )

    st.subheader(
        "Resultados del modelo"
    )

    st.dataframe(

        result["results"],

        use_container_width=True,

        hide_index=True

    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(

            "GWR R²",

            f'{result["r2_gwr"]:.4f}'

        )

    with col2:

        st.metric(

            "GWR RMSE",

            f'{result["rmse_gwr"]:.4f}'

        )

    with col3:

        st.metric(

            "GWR MAE",

            f'{result["mae_gwr"]:.4f}'

        )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(

            "GWRC R²",

            f'{result["r2_gwrc"]:.4f}'

        )

    with col2:

        st.metric(

            "GWRC RMSE",

            f'{result["rmse_gwrc"]:.4f}'

        )

    with col3:

        st.metric(

            "GWRC MAE",

            f'{result["mae_gwrc"]:.4f}'

        )

    st.write(

        f'Bandwidth utilizado: '
        f'**{result["bw_opt"]}**'

    )

    st.write(

        f'CN > 25: '
        f'**{int(np.sum(result["local_CN"] > CN_THRESHOLD))}'
        f'/{len(data)} '
        f'('
        f'{100 * np.sum(result["local_CN"] > CN_THRESHOLD) / len(data):.2f}'
        f'%)'

    )

    st.subheader(
        "VIF global"
    )

    st.dataframe(

        result["vif_table"],

        use_container_width=True,

        hide_index=True

    )

    st.subheader(
        "CN local y lambda local"
    )

    st.dataframe(

        result["cn_table"],

        use_container_width=True,

        hide_index=True

    )

    st.subheader(
        "Coeficientes"
    )

    st.dataframe(

        result["coef_table"],

        use_container_width=True,

        hide_index=True

    )

    st.subheader(
        "Predicciones y residuos"
    )

    st.dataframe(

        result["residual_table"],

        use_container_width=True,

        hide_index=True

    )

    st.subheader(
        "Kriging de residuos GWRC"
    )

    st.write(
        "Residuo utilizado:"
    )

    st.latex(
        r"Residual_{GWRC}=SOC_{observado}-SOC_{GWRC}"
    )

    st.write(
        "Modelo de interpolación: "
        "**Ordinary Kriging con semivariograma spherical**."
    )

    with st.spinner(
        "Generando GWRC, kriging de residuos y GWRCK..."
    ):

        (
            output_paths,
            kriged_residual_values
        ) = create_rasters(

            data,

            result,

            raster_paths

        )

    st.success(
        "GWRC + kriging de residuos + GWRCK generados correctamente."
    )

    st.subheader(
        "Resultados finales"
    )

    residual_gwrc = result[
        "residual_gwrc"
    ]

    residual_kriging_at_points = (
        kriged_residual_values
    )

    gwrck_point = (
        result["gwrc_pred"]
        +
        residual_kriging_at_points
    )

    r2_gwrck = r2_score(
        data["SOC"].values,
        gwrck_point
    )

    rmse_gwrck = np.sqrt(
        mean_squared_error(
            data["SOC"].values,
            gwrck_point
        )
    )

    mae_gwrck = mean_absolute_error(
        data["SOC"].values,
        gwrck_point
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(

            "GWRCK R²",

            f"{r2_gwrck:.4f}"

        )

    with col2:

        st.metric(

            "GWRCK RMSE",

            f"{rmse_gwrck:.4f}"

        )

    with col3:

        st.metric(

            "GWRCK MAE",

            f"{mae_gwrck:.4f}"

        )

    gwrck_metrics = pd.DataFrame({

        "Modelo": [
            "GWR",
            "GWRC",
            "GWRCK"
        ],

        "R2": [
            result["r2_gwr"],
            result["r2_gwrc"],
            r2_gwrck
        ],

        "RMSE": [
            result["rmse_gwr"],
            result["rmse_gwrc"],
            rmse_gwrck
        ],

        "MAE": [
            result["mae_gwr"],
            result["mae_gwrc"],
            mae_gwrck
        ]

    })

    st.dataframe(

        gwrck_metrics,

        use_container_width=True,

        hide_index=True

    )

    excel_buffer = io.BytesIO()

    result["residual_table"][
        "Residual_Kriging"
    ] = (
        residual_kriging_at_points
    )

    result["residual_table"][
        "SOC_GWRCK"
    ] = (
        gwrck_point
    )

    result["residual_table"][
        "Residual_GWRCK"
    ] = (

        data["SOC"].values
        -
        gwrck_point

    )

    with pd.ExcelWriter(

        excel_buffer,

        engine="openpyxl"

    ) as writer:

        result["results"].to_excel(

            writer,

            sheet_name="Modelos",

            index=False

        )

        gwrck_metrics.to_excel(

            writer,

            sheet_name="GWR_GWRC_GWRCK",

            index=False

        )

        result["vif_table"].to_excel(

            writer,

            sheet_name="VIF_Global",

            index=False

        )

        result["cn_table"].to_excel(

            writer,

            sheet_name="CN_Local",

            index=False

        )

        result["coef_table"].to_excel(

            writer,

            sheet_name="Coeficientes",

            index=False

        )

        result["residual_table"].to_excel(

            writer,

            sheet_name="Residuos_GWRC_GWRCK",

            index=False

        )

        result["summary"].to_excel(

            writer,

            sheet_name="Resumen",

            index=False

        )

    st.download_button(

        "Descargar RESULTADOS_GWR_GWRC_GWRCK.xlsx",

        data=excel_buffer.getvalue(),

        file_name=(
            "RESULTADOS_GWR_GWRC_GWRCK.xlsx"
        ),

        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),

        use_container_width=True

    )

    for filename, path in (
        output_paths.items()
    ):

        with open(
            path,
            "rb"
        ) as f:

            st.download_button(

                f"Descargar {filename}",

                data=f.read(),

                file_name=filename,

                mime="image/tiff",

                key=f"download_{filename}",

                use_container_width=True

            )

except Exception as e:

    import traceback

    st.error(

        f"Error durante la ejecución: {e}"

    )

    st.code(

        traceback.format_exc(),

        language="text"

    )
