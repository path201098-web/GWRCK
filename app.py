import os
import warnings
warnings.filterwarnings("ignore")

import io
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
    page_title="GWR-GWRC Spatial Model",
    layout="wide"
)

st.title("GWR-GWRC-GWRCK Spatial Model")
st.write(
    "GWR → GWRC → Kriging de residuos → GWRCK."
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
def save_uploaded_file(uploaded_file):

    import tempfile

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]

    fd, path = tempfile.mkstemp(
        suffix=suffix
    )

    os.close(fd)

    with open(path, "wb") as f:
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
        or bandwidth <= 0
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
            np.isfinite(local_CN[i])
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

                if lambda_local <= 0:

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

    cn_problem = (
        local_CN
        >
        CN_THRESHOLD
    )

    n_cn_problem = np.sum(
        cn_problem
    )

    pct_cn_problem = (
        100
        *
        n_cn_problem
        /
        n
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
            y - gwrc_pred,

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

            n_cn_problem,

            pct_cn_problem,

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

        "gwr_params": gwr_betas,

        "corrected_betas":
            corrected_betas,

        "gwr_pred":
            gwr_pred,

        "gwrc_pred":
            gwrc_pred,

        "gwr_residuals":
            gwr_residuals,

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

    import tempfile

    output_paths = {}

    raster_outputs = {

        "SOC_GWRC_30m.tif":
            gwrc_raster,

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

    return output_paths


def krige_gwrc_residuals(
    data,
    result,
    gwrc_raster_path
):
    """
    Kriging ordinario exclusivamente de los residuos GWRC.

    Residual_GWRC = SOC_Observed - SOC_GWRC

    El GWR y el GWRC no se vuelven a ajustar.
    """

    coords = result["coords"].astype(float)

    residuals = np.asarray(
        result["residual_table"]["Residual_GWRC"],
        dtype=float
    )

    valid_residuals = (
        np.isfinite(coords[:, 0])
        &
        np.isfinite(coords[:, 1])
        &
        np.isfinite(residuals)
    )

    x = coords[valid_residuals, 0]
    y = coords[valid_residuals, 1]
    z = residuals[valid_residuals]

    if len(z) < 3:
        raise ValueError(
            "No existen suficientes residuos GWRC válidos "
            "para realizar el kriging."
        )

    z_mean = float(np.mean(z))
    z_std = float(np.std(z))

    if not np.isfinite(z_std) or z_std <= 1e-12:
        raise ValueError(
            "La desviación estándar de los residuos GWRC "
            "es cero o no es válida."
        )

    # Se mantiene el sistema de coordenadas proyectado EPSG:32717.
    # El escalamiento evita problemas numéricos en las distancias.
    x_center = float(np.mean(x))
    y_center = float(np.mean(y))
    coordinate_scale = 1000.0

    xk = (x - x_center) / coordinate_scale
    yk = (y - y_center) / coordinate_scale

    z_scaled = (
        (z - z_mean)
        / z_std
    )

    with rasterio.open(
        gwrc_raster_path
    ) as src:

        gwrc_raster = src.read(
            1
        ).astype(float)

        transform = src.transform
        profile = src.profile.copy()
        width = src.width
        height = src.height

    # Centros de las celdas del mismo raster GWRC.
    cols = np.arange(
        width,
        dtype=float
    )

    rows = np.arange(
        height,
        dtype=float
    )

    grid_x = (
        transform.c
        +
        (cols + 0.5)
        *
        transform.a
    )

    grid_y = (
        transform.f
        +
        (rows + 0.5)
        *
        transform.e
    )

    grid_x_scaled = (
        grid_x - x_center
    ) / coordinate_scale

    grid_y_scaled = (
        grid_y - y_center
    ) / coordinate_scale

    # PyKrige requiere coordenadas crecientes para la grilla.
    y_reverse = (
        len(grid_y_scaled) > 1
        and grid_y_scaled[0] > grid_y_scaled[-1]
    )

    grid_y_for_kriging = np.sort(
        grid_y_scaled
    )

    ok = OrdinaryKriging(
        xk,
        yk,
        z_scaled,
        variogram_model="spherical",
        nlags=12,
        weight=True,
        verbose=False,
        enable_plotting=False,
        coordinates_type="euclidean"
    )

    with st.spinner(
        "Realizando Kriging de los residuos GWRC..."
    ):

        kriged_scaled, kriging_variance = ok.execute(
            "grid",
            grid_x_scaled,
            grid_y_for_kriging
        )

    if np.ma.isMaskedArray(
        kriged_scaled
    ):

        kriged_scaled = np.ma.filled(
            kriged_scaled,
            np.nan
        )

    if np.ma.isMaskedArray(
        kriging_variance
    ):

        kriging_variance = np.ma.filled(
            kriging_variance,
            np.nan
        )

    kriged_scaled = np.asarray(
        kriged_scaled,
        dtype=float
    )

    kriging_variance = np.asarray(
        kriging_variance,
        dtype=float
    )

    # Regresar a las unidades originales de SOC.
    kriged_residual = (
        kriged_scaled
        *
        z_std
        +
        z_mean
    )

    # La grilla del raster tiene Y descendente.
    if y_reverse:
        kriged_residual = np.flipud(
            kriged_residual
        )

        kriging_variance = np.flipud(
            kriging_variance
        )

    # El GWRCK solamente existe donde existe el raster GWRC.
    valid_gwrc = np.isfinite(
        gwrc_raster
    )

    kriged_residual[
        ~valid_gwrc
    ] = np.nan

    kriging_variance[
        ~valid_gwrc
    ] = np.nan

    if kriged_residual.shape != (
        height,
        width
    ):

        raise ValueError(
            "El raster de residuos krigeados no coincide "
            "con la geometría del raster GWRC."
        )

    # Diagnóstico antes de generar GWRCK.
    finite_kriged = kriged_residual[
        np.isfinite(kriged_residual)
    ]

    if len(finite_kriged) == 0:
        raise ValueError(
            "El Kriging no produjo residuos krigeados válidos."
        )

    # Se controla solamente una explosión numérica extrema.
    # No se reemplazan valores por vecinos ni se recortan.
    observed_abs_max = float(
        np.max(
            np.abs(z)
        )
    )

    kriged_abs_max = float(
        np.max(
            np.abs(finite_kriged)
        )
    )

    if (
        np.isfinite(observed_abs_max)
        and
        np.isfinite(kriged_abs_max)
        and
        kriged_abs_max > max(
            10.0 * observed_abs_max,
            1.0
        )
    ):

        raise ValueError(
            "El Kriging de residuos GWRC produjo valores "
            "extremadamente inestables. "
            f"Máximo residuo observado: "
            f"{observed_abs_max:.6f}; "
            f"máximo krigeado: "
            f"{kriged_abs_max:.6f}."
        )

    # ============================================================
    # GWRCK = GWRC + RESIDUO GWRC KRIGEADO
    # ============================================================

    gwrck_raster = (
        gwrc_raster
        +
        kriged_residual
    )

    # Guardar los rasters con exactamente la misma geometría,
    # CRS y resolución del raster GWRC original.
    output_paths = {}

    profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="lzw"
    )

    import tempfile

    residual_path = os.path.join(
        tempfile.gettempdir(),
        "Residual_GWRC_Kriged_30m.tif"
    )

    variance_path = os.path.join(
        tempfile.gettempdir(),
        "Kriging_Variance_30m.tif"
    )

    gwrck_path = os.path.join(
        tempfile.gettempdir(),
        "SOC_GWRCK_30m.tif"
    )

    with rasterio.open(
        residual_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            kriged_residual.astype(
                np.float32
            ),
            1
        )

    with rasterio.open(
        variance_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            kriging_variance.astype(
                np.float32
            ),
            1
        )

    with rasterio.open(
        gwrck_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            gwrck_raster.astype(
                np.float32
            ),
            1
        )

    output_paths[
        "Residual_GWRC_Kriged_30m.tif"
    ] = residual_path

    output_paths[
        "Kriging_Variance_30m.tif"
    ] = variance_path

    output_paths[
        "SOC_GWRCK_30m.tif"
    ] = gwrck_path

    # ============================================================
    # EVALUACIÓN PUNTUAL DEL GWRCK
    # ============================================================

    point_rows, point_cols = rasterio.transform.rowcol(
        transform,
        coords[:, 0],
        coords[:, 1]
    )

    point_rows = np.asarray(
        point_rows,
        dtype=int
    )

    point_cols = np.asarray(
        point_cols,
        dtype=int
    )

    point_inside = (
        (point_rows >= 0)
        &
        (point_rows < height)
        &
        (point_cols >= 0)
        &
        (point_cols < width)
    )

    residual_kriged_points = np.full(
        len(coords),
        np.nan,
        dtype=float
    )

    residual_kriged_points[
        point_inside
    ] = kriged_residual[
        point_rows[point_inside],
        point_cols[point_inside]
    ]

    observed = data[
        "SOC"
    ].values.astype(float)

    soc_gwrc_points = np.asarray(
        result["gwrc_pred"],
        dtype=float
    )

    soc_gwrck_points = (
        soc_gwrc_points
        +
        residual_kriged_points
    )

    valid_metrics = (
        np.isfinite(observed)
        &
        np.isfinite(soc_gwrck_points)
    )

    n_valid = int(
        np.sum(valid_metrics)
    )

    if n_valid >= 2:

        r2_gwrck = r2_score(
            observed[valid_metrics],
            soc_gwrck_points[
                valid_metrics
            ]
        )

        rmse_gwrck = np.sqrt(
            mean_squared_error(
                observed[valid_metrics],
                soc_gwrck_points[
                    valid_metrics
                ]
            )
        )

        mae_gwrck = mean_absolute_error(
            observed[valid_metrics],
            soc_gwrck_points[
                valid_metrics
            ]
        )

    else:

        r2_gwrck = np.nan
        rmse_gwrck = np.nan
        mae_gwrck = np.nan

    gwrck_results = pd.DataFrame({

        "Modelo": [
            "GWRCK"
        ],

        "N": [
            n_valid
        ],

        "Variables": [
            len(VARS_MODEL)
        ],

        "Bandwidth": [
            result["bw_opt"]
        ],

        "R2_InSample": [
            r2_gwrck
        ],

        "RMSE_InSample": [
            rmse_gwrck
        ],

        "MAE_InSample": [
            mae_gwrck
        ],

        "AIC": [
            np.nan
        ],

        "AICc": [
            np.nan
        ]

    })

    gwrck_point_table = pd.DataFrame({

        "X": data["X"].values,

        "Y": data["Y"].values,

        "SOC_Observed": observed,

        "SOC_GWRC": soc_gwrc_points,

        "Residual_GWRC":
            residuals,

        "Residual_GWRC_Kriged":
            residual_kriged_points,

        "SOC_GWRCK":
            soc_gwrck_points

    })

    diagnostics = pd.DataFrame({

        "Parametro": [

            "Residual_GWRC_min",
            "Residual_GWRC_max",
            "Residual_GWRC_mean",
            "Residual_GWRC_std",
            "Residual_Kriged_min",
            "Residual_Kriged_max",
            "Residual_Kriged_mean",
            "Residual_Kriged_std",
            "N_puntos_GWRCK"

        ],

        "Valor": [

            np.nanmin(z),
            np.nanmax(z),
            np.nanmean(z),
            np.nanstd(z),
            np.nanmin(finite_kriged),
            np.nanmax(finite_kriged),
            np.nanmean(finite_kriged),
            np.nanstd(finite_kriged),
            n_valid

        ]

    })

    return {
        "output_paths": output_paths,
        "gwrck_results": gwrck_results,
        "gwrck_point_table": gwrck_point_table,
        "diagnostics": diagnostics,
        "r2_gwrck": r2_gwrck,
        "rmse_gwrck": rmse_gwrck,
        "mae_gwrck": mae_gwrck,
        "n_valid": n_valid
    }


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

run_model = (
    st.sidebar.button(
        "Ejecutar GWR + GWRC + GWRCK",
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
        "'Ejecutar GWR + GWRC'."

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

    with st.spinner(
        "Generando raster GWRC..."
    ):

        output_paths = create_rasters(

            data,

            result,

            raster_paths

        )

    st.success(
        "Raster GWRC generado correctamente."
    )

    # ============================================================
    # GWRCK
    # ============================================================

    with st.spinner(
        "Generando GWRCK mediante Kriging de residuos GWRC..."
    ):

        gwrck = krige_gwrc_residuals(

            data,

            result,

            output_paths[
                "SOC_GWRC_30m.tif"
            ]

        )

    output_paths.update(
        gwrck["output_paths"]
    )

    st.success(
        "GWRCK generado correctamente."
    )

    st.subheader(
        "Resultados GWRCK"
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "GWRCK R²",
            f'{gwrck["r2_gwrck"]:.4f}'
        )

    with col2:

        st.metric(
            "GWRCK RMSE",
            f'{gwrck["rmse_gwrck"]:.4f}'
        )

    with col3:

        st.metric(
            "GWRCK MAE",
            f'{gwrck["mae_gwrck"]:.4f}'
        )

    st.write(
        f'Puntos válidos para GWRCK: '
        f'**{gwrck["n_valid"]}/{len(data)}**'
    )

    st.info(
        "GWRCK = SOC_GWRC + residuo GWRC krigeado. "
        "El GWR y el GWRC originales no se vuelven a ajustar."
    )

    st.subheader(
        "Predicciones puntuales GWRCK"
    )

    st.dataframe(
        gwrck["gwrck_point_table"],
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Diagnóstico de residuos"
    )

    st.dataframe(
        gwrck["diagnostics"],
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Descargas"
    )

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(

        excel_buffer,

        engine="openpyxl"

    ) as writer:

        result["results"].to_excel(

            writer,

            sheet_name="Modelos",

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

            sheet_name="Predicciones_GWRC",

            index=False

        )

        result["summary"].to_excel(

            writer,

            sheet_name="Resumen",

            index=False

        )

        gwrck["gwrck_results"].to_excel(

            writer,

            sheet_name="Modelo_GWRCK",

            index=False

        )

        gwrck["gwrck_point_table"].to_excel(

            writer,

            sheet_name="Predicciones_GWRCK",

            index=False

        )

        gwrck["diagnostics"].to_excel(

            writer,

            sheet_name="Diagnostico_Kriging",

            index=False

        )

    st.download_button(

        "Descargar RESULTADOS_GWR_GWRC_GWRCK.xlsx",

        data=excel_buffer.getvalue(),

        file_name="RESULTADOS_GWR_GWRC_GWRCK.xlsx",

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
