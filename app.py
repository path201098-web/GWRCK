import streamlit as st
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from scipy.interpolate import griddata
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from pykrige.ok import OrdinaryKriging
import tempfile
import os
import zipfile
import io

st.set_page_config(
    page_title="GWRCK",
    layout="wide"
)

st.title("GWR → GWRC → Kriging de residuos → GWRCK")

st.sidebar.header("Configuración")

bandwidth_manual = st.sidebar.number_input(
    "Bandwidth",
    min_value=2,
    max_value=200,
    value=57,
    step=1
)

automatic_bw = st.sidebar.checkbox(
    "Calcular bandwidth automáticamente mediante AICc",
    value=False
)

ridge_min_lambda = 0.001

uploaded_data = st.file_uploader(
    "Subir archivo de datos CSV o Excel",
    type=["csv", "xlsx", "xls"]
)

uploaded_rasters = st.file_uploader(
    "Subir los 8 rasters predictivos en formato GeoTIFF",
    type=["tif", "tiff"],
    accept_multiple_files=True
)

st.info(
    "Variables requeridas: SOC, X, Y, ELEV, Curvatu, Slope, "
    "HillSha, TEMP, PRES, EVI y BSI."
)

required_columns = [
    "SOC",
    "X",
    "Y",
    "ELEV",
    "Curvatu",
    "Slope",
    "HillSha",
    "TEMP",
    "PRES",
    "EVI",
    "BSI"
]

predictor_names = [
    "ELEV",
    "Curvatu",
    "Slope",
    "HillSha",
    "TEMP",
    "PRES",
    "EVI",
    "BSI"
]


def read_input_data(uploaded_file):

    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = df.columns.astype(str).str.strip()

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Faltan las siguientes columnas: "
            + ", ".join(missing)
        )

    df = df[required_columns].copy()

    for col in required_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna().reset_index(drop=True)

    return df


def calculate_cn_and_ridge(X_local, y_local):

    X_local = np.asarray(X_local, dtype=float)
    y_local = np.asarray(y_local, dtype=float).reshape(-1, 1)

    X_design = np.column_stack(
        [
            np.ones(X_local.shape[0]),
            X_local
        ]
    )

    try:

        singular_values = np.linalg.svd(
            X_design,
            compute_uv=False
        )

        if singular_values[-1] <= 0:

            cn = np.inf

        else:

            cn = (
                singular_values[0]
                /
                singular_values[-1]
            )

    except Exception:

        cn = np.inf

    if cn <= 25:

        beta = np.linalg.lstsq(
            X_design,
            y_local,
            rcond=None
        )[0]

        lambda_local = 0.0

        return (
            beta.flatten(),
            cn,
            lambda_local
        )

    XtX = X_design.T @ X_design

    eigvals = np.linalg.eigvalsh(
        XtX
    )

    eig_min = np.min(eigvals)

    if eig_min > 0:

        lambda_local = (
            0.01 * eig_min
        )

    else:

        lambda_local = ridge_min_lambda

    lambda_local = max(
        lambda_local,
        ridge_min_lambda
    )

    penalty = np.eye(
        X_design.shape[1]
    )

    penalty[0, 0] = 0.0

    beta_ridge = np.linalg.solve(
        XtX + lambda_local * penalty,
        X_design.T @ y_local
    )

    return (
        beta_ridge.flatten(),
        cn,
        lambda_local
    )


def local_gwrc_coefficients(
    coords,
    X,
    y,
    gwr_res
):

    n = len(y)

    corrected_betas = np.zeros(
        (n, X.shape[1] + 1)
    )

    cn_values = np.zeros(n)

    lambda_values = np.zeros(n)

    gwr_betas = np.asarray(
        gwr_res.params
    )

    for i in range(n):

        distances = np.sqrt(
            np.sum(
                (coords - coords[i]) ** 2,
                axis=1
            )
        )

        order = np.argsort(
            distances
        )

        bw = min(
            bandwidth_used,
            len(order)
        )

        neighbors = order[:bw]

        d = distances[
            neighbors
        ]

        max_d = np.max(d)

        if max_d == 0:

            weights = np.ones(
                len(neighbors)
            )

        else:

            u = d / max_d

            weights = (
                1 - u ** 2
            ) ** 2

        W = np.diag(weights)

        X_design = np.column_stack(
            [
                np.ones(
                    len(neighbors)
                ),
                X[neighbors]
            ]
        )

        y_local = y[
            neighbors
        ].reshape(-1, 1)

        XtWX = (
            X_design.T
            @ W
            @ X_design
        )

        try:

            eigvals = np.linalg.eigvalsh(
                XtWX
            )

            eigvals = np.maximum(
                eigvals,
                0
            )

            if eigvals[-1] == 0:

                cn = np.inf

            elif eigvals[0] == 0:

                cn = np.inf

            else:

                cn = np.sqrt(
                    eigvals[-1]
                    /
                    eigvals[0]
                )

        except Exception:

            cn = np.inf

        cn_values[i] = cn

        if cn <= 25:

            corrected_betas[i] = (
                gwr_betas[i]
            )

            lambda_values[i] = 0.0

        else:

            eigvals = np.linalg.eigvalsh(
                XtWX
            )

            eig_min = np.min(
                eigvals
            )

            if eig_min > 0:

                lambda_local = (
                    0.01 * eig_min
                )

            else:

                lambda_local = ridge_min_lambda

            lambda_local = max(
                lambda_local,
                ridge_min_lambda
            )

            penalty = np.eye(
                X_design.shape[1]
            )

            penalty[0, 0] = 0.0

            beta_ridge = np.linalg.solve(
                XtWX
                + lambda_local * penalty,
                X_design.T
                @ W
                @ y_local
            )

            corrected_betas[i, 0] = (
                beta_ridge[0, 0]
            )

            corrected_betas[i, 1:] = (
                beta_ridge[1:, 0]
            )

            lambda_values[i] = (
                lambda_local
            )

    return (
        corrected_betas,
        cn_values,
        lambda_values
    )


def create_prediction_raster(
    src,
    predictor_rasters,
    predictor_names,
    point_coords,
    local_betas,
    output_path
):

    height = src.height
    width = src.width

    transform = src.transform

    crs = src.crs

    rows, cols = np.indices(
        (height, width)
    )

    xs, ys = rasterio.transform.xy(
        transform,
        rows,
        cols
    )

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    grid_points = np.column_stack(
        [
            xs.ravel(),
            ys.ravel()
        ]
    )

    beta_grid = []

    for j in range(
        local_betas.shape[1]
    ):

        beta_interp = griddata(
            point_coords,
            local_betas[:, j],
            grid_points,
            method="nearest"
        )

        beta_grid.append(
            beta_interp
        )

    beta_grid = np.asarray(
        beta_grid
    )

    prediction = (
        beta_grid[0]
    )

    for j, name in enumerate(
        predictor_names,
        start=1
    ):

        raster_array = (
            predictor_rasters[name]
        )

        values = raster_array.ravel()

        prediction += (
            beta_grid[j]
            * values
        )

    prediction = prediction.reshape(
        height,
        width
    )

    profile = src.profile.copy()

    profile.update(
        dtype="float32",
        count=1,
        compress="lzw",
        nodata=np.nan
    )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            prediction.astype(
                "float32"
            ),
            1
        )

    return prediction


def save_raster(
    array,
    reference,
    output_path
):

    profile = reference.profile.copy()

    profile.update(
        dtype="float32",
        count=1,
        compress="lzw",
        nodata=np.nan
    )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            np.asarray(
                array,
                dtype="float32"
            ),
            1
        )


if st.button(
    "Ejecutar GWR → GWRC → GWRCK",
    type="primary"
):

    if uploaded_data is None:

        st.error(
            "Debe subir el archivo de datos."
        )

        st.stop()

    if (
        uploaded_rasters is None
        or len(uploaded_rasters) != 8
    ):

        st.error(
            "Debe subir exactamente los 8 rasters predictivos."
        )

        st.stop()

    try:

        df = read_input_data(
            uploaded_data
        )

        st.write(
            f"Número de observaciones: {len(df)}"
        )

        if len(df) < 10:

            st.error(
                "El número de observaciones es insuficiente."
            )

            st.stop()

        y = df["SOC"].values.astype(
            float
        )

        y_gwr = y.reshape(
            (-1, 1)
        )

        coords_original = df[
            ["X", "Y"]
        ].values.astype(
            float
        )

        X = df[
            predictor_names
        ].values.astype(
            float
        )

        st.write(
            "Ajustando GWR..."
        )

        selector = Sel_BW(
            coords_original,
            y_gwr,
            X,
            kernel="bisquare",
            fixed=False,
            constant=True
        )

        if automatic_bw:

            bw_opt = selector.search(
                criterion="AICc"
            )

        else:

            bw_opt = bandwidth_manual

        bandwidth_used = int(
            round(bw_opt)
        )

        st.write(
            f"Bandwidth utilizado: {bandwidth_used}"
        )

        gwr_model = GWR(
            coords_original,
            y_gwr,
            X,
            bw=bandwidth_used,
            fixed=False,
            kernel="bisquare",
            constant=True
        )

        gwr_res = gwr_model.fit()

        gwr_pred = np.asarray(
            gwr_res.predy
        ).flatten()

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

        st.success(
            "GWR calculado correctamente."
        )

        st.write(
            f"GWR R²: {r2_gwr:.6f}"
        )

        st.write(
            f"GWR RMSE: {rmse_gwr:.6f}"
        )

        st.write(
            f"GWR MAE: {mae_gwr:.6f}"
        )

        st.write(
            "Calculando corrección GWRC..."
        )

        (
            corrected_betas,
            cn_values,
            lambda_values
        ) = local_gwrc_coefficients(
            coords_original,
            X,
            y,
            gwr_res
        )

        X_design_all = np.column_stack(
            [
                np.ones(
                    len(X)
                ),
                X
            ]
        )

        gwrc_pred = np.sum(
            X_design_all
            * corrected_betas,
            axis=1
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

        st.success(
            "GWRC calculado correctamente."
        )

        st.write(
            f"GWRC R²: {r2_gwrc:.6f}"
        )

        st.write(
            f"GWRC RMSE: {rmse_gwrc:.6f}"
        )

        st.write(
            f"GWRC MAE: {mae_gwrc:.6f}"
        )

        st.write(
            "Calculando residuos de GWRC..."
        )

        residual_gwrc = (
            y - gwrc_pred
        )

        st.write(
            "Preparando rasters..."
        )

        raster_dict = {}

        reference_src = None

        for uploaded_raster in uploaded_rasters:

            name = os.path.splitext(
                uploaded_raster.name
            )[0]

            matched_name = None

            for predictor in predictor_names:

                if (
                    name.lower()
                    ==
                    predictor.lower()
                    or
                    predictor.lower()
                    in name.lower()
                ):

                    matched_name = predictor

                    break

            if matched_name is None:

                continue

            temp_raster = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".tif"
            )

            temp_raster.write(
                uploaded_raster.getbuffer()
            )

            temp_raster.close()

            src = rasterio.open(
                temp_raster.name
            )

            array = src.read(
                1
            ).astype(
                float
            )

            raster_dict[
                matched_name
            ] = array

            if reference_src is None:

                reference_src = src

        missing_rasters = [
            p for p in predictor_names
            if p not in raster_dict
        ]

        if missing_rasters:

            st.error(
                "No se encontraron los rasters: "
                + ", ".join(
                    missing_rasters
                )
            )

            st.stop()

        transformer = Transformer.from_crs(
            "EPSG:4326",
            reference_src.crs,
            always_xy=True
        )

        point_x, point_y = transformer.transform(
            coords_original[:, 0],
            coords_original[:, 1]
        )

        point_coords = np.column_stack(
            [
                point_x,
                point_y
            ]
        )

        output_dir = tempfile.mkdtemp()

        soc_gwrc_path = os.path.join(
            output_dir,
            "SOC_GWRC_30m.tif"
        )

        st.write(
            "Generando raster GWRC..."
        )

        gwrc_raster = create_prediction_raster(
            reference_src,
            raster_dict,
            predictor_names,
            point_coords,
            corrected_betas,
            soc_gwrc_path
        )

        st.success(
            "Raster GWRC generado."
        )

        st.write(
            "Ejecutando Kriging de residuos GWRC..."
        )

        ok = OrdinaryKriging(
            point_x,
            point_y,
            residual_gwrc,
            variogram_model="spherical",
            nlags=12,
            weight=False,
            exact_values=True,
            verbose=False
        )

        height = reference_src.height
        width = reference_src.width

        rows, cols = np.indices(
            (height, width)
        )

        grid_x, grid_y = rasterio.transform.xy(
            reference_src.transform,
            rows,
            cols
        )

        grid_x = np.asarray(
            grid_x
        )

        grid_y = np.asarray(
            grid_y
        )

        kriged_residual, kriging_variance = ok.execute(
            "points",
            grid_x.ravel(),
            grid_y.ravel()
        )

        residual_kriging_raster = np.asarray(
            kriged_residual
        ).reshape(
            height,
            width
        )

        residual_kriging_path = os.path.join(
            output_dir,
            "Residual_GWRC_Kriging_30m.tif"
        )

        save_raster(
            residual_kriging_raster,
            reference_src,
            residual_kriging_path
        )

        st.success(
            "Kriging de residuos generado."
        )

        st.write(
            "Ejecutando calculadora raster..."
        )

        soc_gwrck_raster = (
            gwrc_raster
            +
            residual_kriging_raster
        )

        soc_gwrck_path = os.path.join(
            output_dir,
            "SOC_GWRCK_30m.tif"
        )

        save_raster(
            soc_gwrck_raster,
            reference_src,
            soc_gwrck_path
        )

        st.success(
            "GWRCK generado correctamente."
        )

        st.write(
            "Extrayendo SOC_GWRCK del raster final..."
        )

        rows_points, cols_points = rowcol(
            reference_src.transform,
            point_x,
            point_y
        )

        gwrck_points = []

        gwrc_raster_points = []

        kriging_raster_points = []

        for r, c in zip(
            rows_points,
            cols_points
        ):

            if (
                r < 0
                or r >= height
                or c < 0
                or c >= width
            ):

                gwrck_points.append(
                    np.nan
                )

                gwrc_raster_points.append(
                    np.nan
                )

                kriging_raster_points.append(
                    np.nan
                )

            else:

                gwrck_points.append(
                    soc_gwrck_raster[
                        r,
                        c
                    ]
                )

                gwrc_raster_points.append(
                    gwrc_raster[
                        r,
                        c
                    ]
                )

                kriging_raster_points.append(
                    residual_kriging_raster[
                        r,
                        c
                    ]
                )

        gwrck_points = np.asarray(
            gwrck_points
        )

        valid = (
            np.isfinite(y)
            &
            np.isfinite(gwrck_points)
        )

        y_eval = y[
            valid
        ]

        gwrck_eval = gwrck_points[
            valid
        ]

        r2_gwrck = r2_score(
            y_eval,
            gwrck_eval
        )

        rmse_gwrck = np.sqrt(
            mean_squared_error(
                y_eval,
                gwrck_eval
            )
        )

        mae_gwrck = mean_absolute_error(
            y_eval,
            gwrck_eval
        )

        st.success(
            "Evaluación del raster GWRCK final completada."
        )

        st.subheader(
            "Resultados"
        )

        results = pd.DataFrame(
            {
                "Modelo": [
                    "GWR",
                    "GWRC",
                    "GWRCK"
                ],
                "R2": [
                    r2_gwr,
                    r2_gwrc,
                    r2_gwrck
                ],
                "RMSE": [
                    rmse_gwr,
                    rmse_gwrc,
                    rmse_gwrck
                ],
                "MAE": [
                    mae_gwr,
                    mae_gwrc,
                    mae_gwrck
                ]
            }
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        st.write(
            f"GWRCK R²: {r2_gwrck:.6f}"
        )

        st.write(
            f"GWRCK RMSE: {rmse_gwrck:.6f}"
        )

        st.write(
            f"GWRCK MAE: {mae_gwrck:.6f}"
        )

        results_points = df.copy()

        results_points[
            "SOC_GWR"
        ] = gwr_pred

        results_points[
            "SOC_GWRC"
        ] = gwrc_pred

        results_points[
            "Residual_GWRC"
        ] = residual_gwrc

        results_points[
            "Residual_Kriging"
        ] = kriging_raster_points

        results_points[
            "SOC_GWRCK"
        ] = gwrck_points

        results_points[
            "CN_Local"
        ] = cn_values

        results_points[
            "Lambda_Local"
        ] = lambda_values

        summary = pd.DataFrame(
            {
                "Modelo": [
                    "GWR",
                    "GWRC",
                    "GWRCK"
                ],
                "R2": [
                    r2_gwr,
                    r2_gwrc,
                    r2_gwrck
                ],
                "RMSE": [
                    rmse_gwr,
                    rmse_gwrc,
                    rmse_gwrck
                ],
                "MAE": [
                    mae_gwr,
                    mae_gwrc,
                    mae_gwrck
                ],
                "Bandwidth": [
                    bandwidth_used,
                    bandwidth_used,
                    bandwidth_used
                ]
            }
        )

        excel_path = os.path.join(
            output_dir,
            "RESULTADOS_GWR_GWRC_GWRCK.xlsx"
        )

        with pd.ExcelWriter(
            excel_path,
            engine="openpyxl"
        ) as writer:

            summary.to_excel(
                writer,
                sheet_name="Metricas",
                index=False
            )

            results_points.to_excel(
                writer,
                sheet_name="Predicciones",
                index=False
            )

            pd.DataFrame(
                {
                    "CN_Local": cn_values,
                    "Lambda_Local": lambda_values
                }
            ).to_excel(
                writer,
                sheet_name="Diagnostico_GWRC",
                index=False
            )

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(
            zip_buffer,
            "w",
            zipfile.ZIP_DEFLATED
        ) as zip_file:

            zip_file.write(
                soc_gwrc_path,
                "SOC_GWRC_30m.tif"
            )

            zip_file.write(
                residual_kriging_path,
                "Residual_GWRC_Kriging_30m.tif"
            )

            zip_file.write(
                soc_gwrck_path,
                "SOC_GWRCK_30m.tif"
            )

            zip_file.write(
                excel_path,
                "RESULTADOS_GWR_GWRC_GWRCK.xlsx"
            )

        zip_buffer.seek(0)

        st.download_button(
            label="Descargar resultados completos",
            data=zip_buffer,
            file_name="GWR_GWRC_GWRCK_resultados.zip",
            mime="application/zip"
        )

        st.download_button(
            label="Descargar Excel",
            data=open(
                excel_path,
                "rb"
            ).read(),
            file_name="RESULTADOS_GWR_GWRC_GWRCK.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:

        st.error(
            "Se produjo un error durante la ejecución."
        )

        st.exception(e)
