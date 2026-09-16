import os
from pathlib import Path

import numpy as np
import streamlit as st
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform as rio_transform
)

import folium
from folium.raster_layers import ImageOverlay
from branca.colormap import LinearColormap
from streamlit_folium import st_folium


st.set_page_config(
    page_title="SOC Viewer | Amoju River Valley",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# -----------------------------------------------------------------------------
# CONFIGURACIÓN DEL VISOR
# -----------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

GWRC_FILE = DATA_DIR / "SOC_GWRC_30m.tif"
GWRCK_FILE = DATA_DIR / "SOC_GWRCK_30m.tif"

MAX_WEB_DIM = 1400
MAP_HEIGHT = 700


# -----------------------------------------------------------------------------
# ESTILO DE LA INTERFAZ
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>

        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 1rem;
        }

        .context-card {
            background: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 12px;
            padding: 1.05rem 1.15rem;
            margin: 0.5rem 0 1rem 0;
            line-height: 1.58;
        }

        .context-card b {
            color: var(--text-color);
        }

        .section-card {
            color: var(--text-color);
            border-left: 4px solid rgba(128, 128, 128, 0.65);
            padding: 0.15rem 0 0.15rem 0.9rem;
            margin: 1rem 0;
        }

        .section-card b {
            color: var(--text-color);
        }

        .query-card {
            background: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-top: 0.75rem;
        }

        .query-panel-title {
            color: var(--text-color);
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0.25rem 0 0.15rem 0;
        }

        .query-title {
            color: var(--text-color);
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .map-label {
            color: var(--text-color);
            font-weight: 650;
            margin-bottom: 0.35rem;
        }

        .footer-note {
            color: var(--text-color);
            opacity: 0.6;
            text-align: center;
            font-size: 0.82rem;
            margin-top: 1.2rem;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# FUNCIONES DE VISUALIZACIÓN
# -----------------------------------------------------------------------------

def _continuous_rgba(values, vmin, vmax):

    values = np.asarray(values, dtype=float)

    normalized = (
        (values - vmin) /
        (vmax - vmin)
        if vmax > vmin
        else np.zeros_like(values)
    )

    normalized = np.clip(normalized, 0, 1)

    stops = np.array([
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37]
    ], dtype=float)

    positions = np.linspace(0, 1, len(stops))

    rgba = np.zeros(
        values.shape + (4,),
        dtype=np.uint8
    )

    finite = np.isfinite(values)

    for channel in range(3):
        rgba[..., channel] = np.interp(
            normalized,
            positions,
            stops[:, channel]
        ).astype(np.uint8)

    rgba[..., 3] = np.where(
        finite,
        205,
        0
    ).astype(np.uint8)

    return rgba


def _read_raster_for_webmap(path):

    with rasterio.open(path) as src:

        data = src.read(1).astype(float)

        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        transform, width, height = calculate_default_transform(
            src.crs,
            "EPSG:4326",
            src.width,
            src.height,
            *src.bounds
        )

        if max(width, height) > MAX_WEB_DIM:

            scale = MAX_WEB_DIM / max(width, height)

            width = max(
                1,
                int(width * scale)
            )

            height = max(
                1,
                int(height * scale)
            )

            transform, width, height = calculate_default_transform(
                src.crs,
                "EPSG:4326",
                src.width,
                src.height,
                *src.bounds,
                dst_width=width,
                dst_height=height
            )

        destination = np.full(
            (height, width),
            np.nan,
            dtype=np.float32
        )

        reproject(
            source=data,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear
        )

    finite = np.isfinite(destination)

    if not np.any(finite):
        raise ValueError(
            f"No valid raster values were found in {path}."
        )

    values = destination[finite]

    vmin = float(
        np.nanpercentile(values, 2)
    )

    vmax = float(
        np.nanpercentile(values, 98)
    )

    rgba = _continuous_rgba(
        destination,
        vmin,
        vmax
    )

    left = transform.c
    top = transform.f

    right = (
        left +
        transform.a * width
    )

    bottom = (
        top +
        transform.e * height
    )

    bounds = [
        [bottom, left],
        [top, right]
    ]

    return (
        rgba,
        bounds,
        vmin,
        vmax
    )


def _common_scale(
    raster_a,
    raster_b
):

    with rasterio.open(raster_a) as src_a:
        a = src_a.read(1).astype(float)

        if src_a.nodata is not None:
            a[a == src_a.nodata] = np.nan

    with rasterio.open(raster_b) as src_b:
        b = src_b.read(1).astype(float)

        if src_b.nodata is not None:
            b[b == src_b.nodata] = np.nan

    values = np.concatenate(
        [
            a[np.isfinite(a)],
            b[np.isfinite(b)]
        ]
    )

    if values.size == 0:
        raise ValueError(
            "No valid values were found in the two rasters."
        )

    vmin = float(
        np.nanpercentile(values, 2)
    )

    vmax = float(
        np.nanpercentile(values, 98)
    )

    return vmin, vmax


def _query_raster_value(
    path,
    lon,
    lat
):

    with rasterio.open(path) as src:

        xs, ys = rio_transform(
            "EPSG:4326",
            src.crs,
            [lon],
            [lat]
        )

        x = xs[0]
        y = ys[0]

        row, col = src.index(
            x,
            y
        )

        if (
            row < 0 or
            row >= src.height or
            col < 0 or
            col >= src.width
        ):
            return None, None, None

        value = src.read(
            1,
            window=(
                (row, row + 1),
                (col, col + 1)
            )
        )[0, 0]

        if src.nodata is not None:
            if np.isclose(
                value,
                src.nodata
            ):
                return None, row, col

        if not np.isfinite(value):
            return None, row, col

        return (
            float(value),
            row,
            col
        )


def create_soc_map(
    gwrc_path,
    gwrck_path
):

    gwrc_rgba, gwrc_bounds, _, _ = (
        _read_raster_for_webmap(
            gwrc_path
        )
    )

    gwrck_rgba, gwrck_bounds, _, _ = (
        _read_raster_for_webmap(
            gwrck_path
        )
    )

    all_bounds = [
        gwrc_bounds,
        gwrck_bounds
    ]

    min_lat = min(
        b[0][0]
        for b in all_bounds
    )

    max_lat = max(
        b[1][0]
        for b in all_bounds
    )

    min_lon = min(
        b[0][1]
        for b in all_bounds
    )

    max_lon = max(
        b[1][1]
        for b in all_bounds
    )

    center_lat = (
        min_lat +
        max_lat
    ) / 2

    center_lon = (
        min_lon +
        max_lon
    ) / 2

    vmin, vmax = _common_scale(
        gwrc_path,
        gwrck_path
    )

    m = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=12,
        control_scale=True,
        tiles=None
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite imagery",
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
        image=gwrc_rgba,
        bounds=gwrc_bounds,
        opacity=0.72,
        interactive=False,
        cross_origin=False,
        zindex=2,
        name="GWRC"
    ).add_to(m)

    ImageOverlay(
        image=gwrck_rgba,
        bounds=gwrck_bounds,
        opacity=0.78,
        interactive=False,
        cross_origin=False,
        zindex=3,
        name="GWRCK"
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
        caption="SOC (Mg ha⁻¹)"
    )

    colormap.add_to(m)

    halo_css = """
    <style>
        .legend text,
        .legend label,
        .legend div {
            paint-order: stroke fill;
            stroke: white;
            stroke-width: 3px;
            stroke-linejoin: round;
            text-shadow:
                0 0 3px white,
                0 0 3px white;
            font-weight: 600;
        }
    </style>
    """

    m.get_root().html.add_child(
        folium.Element(
            halo_css
        )
    )

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    return m


# -----------------------------------------------------------------------------
# COMPROBACIÓN DE LOS RESULTADOS PRECALCULADOS
# -----------------------------------------------------------------------------

missing_files = []

if not GWRC_FILE.exists():
    missing_files.append(
        GWRC_FILE.name
    )

if not GWRCK_FILE.exists():
    missing_files.append(
        GWRCK_FILE.name
    )

if missing_files:

    st.error(
        "The following result files required by the viewer were not found: "
        + ", ".join(missing_files)
    )

    st.info(
        "Place the final GeoTIFF files inside the project's 'data' folder "
        "before deploying the application. End users do not need to upload "
        "any files."
    )

    st.stop()


# -----------------------------------------------------------------------------
# INTERFAZ PRINCIPAL
# -----------------------------------------------------------------------------

left_col, right_col = st.columns(
    [0.38, 0.62],
    gap="large"
)


# -----------------------------------------------------------------------------
# PANEL IZQUIERDO
# -----------------------------------------------------------------------------

with left_col:

    st.markdown(
        "## Soil Organic Carbon Content and Spatial Distribution "
        "in the Amoju River Valley"
    )

    st.caption(
        "Amoju River Valley, Jaen, Peru"
    )

    st.markdown(
        """
        <div class="context-card">
        The Amoju River Valley in northwestern Peru is an important agricultural
        area where rice cultivation represents a key productive resource.
        Soil organic carbon is an important component of soil functioning because
        its spatial distribution is related to soil quality, nutrient dynamics,
        and the capacity of agricultural soils to retain and cycle carbon.

        <br><br>

        This interactive map presents the spatial distribution of soil organic
        carbon estimated using the <b>GWRC</b> and <b>GWRCK</b> models. The
        resulting spatial information can help identify patterns and areas with
        contrasting soil carbon content across the agricultural landscape and
        support the planning of soil conservation and sustainable soil
        management strategies for rice production in the Amoju River Valley.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-card"><b>How to use the viewer</b></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        1. Use the **Layer control** on the map to switch between GWRC and GWRCK.
        2. Zoom to the area of interest.
        3. **Click on any pixel** to retrieve its SOC value.
        4. The value is read from the original 30 × 30 m GeoTIFF.
        """
    )

    st.markdown(
        '<div class="section-card"><b>Spatial models</b></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        **GWRC** — Geographically Weighted Regression with local ridge correction.  

        **GWRCK** — GWRC combined with kriged GWRC residuals.
        """
    )


# -----------------------------------------------------------------------------
# PANEL DERECHO
# -----------------------------------------------------------------------------

with right_col:

    st.markdown(
        '<div class="map-label">Interactive SOC spatial distribution</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Click on the map to retrieve SOC for the corresponding "
        "30 × 30 m pixel. Both models use a common color scale."
    )

    with st.spinner(
        "Loading SOC map..."
    ):

        soc_map = create_soc_map(
            GWRC_FILE,
            GWRCK_FILE
        )

    map_data = st_folium(
        soc_map,
        width=None,
        height=MAP_HEIGHT,
        returned_objects=[
            "last_clicked"
        ]
    )

    click_data = (
        map_data.get(
            "last_clicked"
        )
        if map_data
        else None
    )


    # -------------------------------------------------------------------------
    # INFORMACIÓN DEL PÍXEL
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="query-panel-title">Pixel information</div>',
        unsafe_allow_html=True
    )

    with st.container():

        if click_data:

            lat = float(
                click_data["lat"]
            )

            lon = float(
                click_data["lng"]
            )

            gwrc_value, gwrc_row, gwrc_col = (
                _query_raster_value(
                    GWRC_FILE,
                    lon,
                    lat
                )
            )

            gwrck_value, _, _ = (
                _query_raster_value(
                    GWRCK_FILE,
                    lon,
                    lat
                )
            )

            pixel_lat = lat
            pixel_lon = lon

            if gwrc_value is not None:

                with rasterio.open(
                    GWRC_FILE
                ) as src:

                    xs, ys = rio_transform(
                        "EPSG:4326",
                        src.crs,
                        [lon],
                        [lat]
                    )

                    row, col = src.index(
                        xs[0],
                        ys[0]
                    )

                    cx, cy = rasterio.transform.xy(
                        src.transform,
                        row,
                        col,
                        offset="center"
                    )

                    plon, plat = rio_transform(
                        src.crs,
                        "EPSG:4326",
                        [cx],
                        [cy]
                    )

                    pixel_lon = float(
                        plon[0]
                    )

                    pixel_lat = float(
                        plat[0]
                    )

            st.markdown(
                '<div class="query-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="query-title">'
                'SOC Value of the Selected Pixel'
                '</div>',
                unsafe_allow_html=True
            )

            st.caption(
                f"Pixel center: "
                f"{pixel_lat:.6f}°, "
                f"{pixel_lon:.6f}°"
            )

            q1, q2 = st.columns(2)

            with q1:

                st.metric(
                    "GWRC",
                    (
                        f"{gwrc_value:.2f} Mg ha⁻¹"
                        if gwrc_value is not None
                        else "No data"
                    )
                )

            with q2:

                st.metric(
                    "GWRCK",
                    (
                        f"{gwrck_value:.2f} Mg ha⁻¹"
                        if gwrck_value is not None
                        else "No data"
                    )
                )

            if (
                gwrc_row is not None
                and
                gwrc_col is not None
            ):

                st.caption(
                    f"GWRC raster cell: "
                    f"row {gwrc_row + 1}, "
                    f"column {gwrc_col + 1}"
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.info(
                "Click on a pixel in the map to display "
                "its GWRC and GWRCK SOC values."
            )


# -----------------------------------------------------------------------------
# INFORMACIÓN ADICIONAL
# -----------------------------------------------------------------------------

st.divider()

st.markdown(
    """
    **Data interpretation**  

    SOC values are expressed as **Mg ha⁻¹**. Each pixel represents a
    30 × 30 m spatial unit. The displayed model results were previously
    processed and are provided directly through this viewer; no model
    recalculation is performed by the application.
    """
)

st.markdown(
    """
    <div class="footer-note">
        Soil Organic Carbon Viewer · GWRC and GWRCK spatial models ·
        Amoju River Valley, Jaen, Peru
    </div>
    """,
    unsafe_allow_html=True
)
