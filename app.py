import os
from pathlib import Path

import numpy as np
import streamlit as st
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform as rio_transform

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

        .hero-title {
            font-size: 2.35rem;
            line-height: 1.12;
            font-weight: 700;
            margin: 0 0 0.45rem 0;
        }

        .hero-location {
            color: #667085;
            font-size: 1rem;
            margin-bottom: 1.25rem;
        }

        .context-card {
            background: #f6f8fa;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.05rem 1.15rem;
            margin: 0.5rem 0 1rem 0;
            line-height: 1.58;
        }

        .section-card {
            border-left: 4px solid #6b7280;
            padding: 0.15rem 0 0.15rem 0.9rem;
            margin: 1rem 0;
        }

        .query-card {
            background: #f8fafc;
            border: 1px solid #dbe3ea;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-top: 0.75rem;
        }

        .query-title {
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .map-label {
            font-weight: 650;
            margin-bottom: 0.35rem;
        }

        .footer-note {
            text-align: center;
            color: #6b7280;
            font-size: 0.82rem;
            margin-top: 1.2rem;
        }

        @media (max-width: 900px) {
            .hero-title {
                font-size: 1.8rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# FUNCIONES DE VISUALIZACIÓN
# -----------------------------------------------------------------------------


def _continuous_rgba(values, vmin, vmax):
    """Convierte un raster continuo en RGBA únicamente para visualización web."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)

    rgba = np.zeros(arr.shape + (4,), dtype=np.uint8)

    if not np.any(finite):
        return rgba

    if vmax <= vmin:
        vmax = vmin + 1.0

    t = np.clip(
        (arr - vmin) / (vmax - vmin),
        0.0,
        1.0
    )

    # Paleta continua tipo viridis.
    stops = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    colors = np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37]
        ],
        dtype=float
    )

    for channel in range(3):
        rgba[..., channel] = np.interp(
            t,
            stops,
            colors[:, channel]
        ).astype(np.uint8)

    rgba[..., 3] = np.where(
        finite,
        205,
        0
    ).astype(np.uint8)

    return rgba


def _read_raster_for_webmap(path):
    """Lee un GeoTIFF y lo reproyecta únicamente para su visualización web."""

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
                f"El raster {path.name} no contiene valores válidos para visualizar."
            )

        scale = max(data.shape) / MAX_WEB_DIM

        if scale > 1:
            dst_width = max(
                1,
                int(data.shape[1] / scale)
            )
            dst_height = max(
                1,
                int(data.shape[0] / scale)
            )
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

    if not np.any(finite_dst):
        raise ValueError(
            f"El raster {path.name} no contiene valores válidos después de la reproyección."
        )

    values = destination[finite_dst]

    left = dst_transform.c
    top = dst_transform.f
    right = left + dst_transform.a * dst_width
    bottom = top + dst_transform.e * dst_height

    bounds = [
        [bottom, left],
        [top, right]
    ]

    return destination, bounds


def _common_scale(raster_a, raster_b):
    """Obtiene una escala común para comparar GWRC y GWRCK."""

    values_a = raster_a[np.isfinite(raster_a)]
    values_b = raster_b[np.isfinite(raster_b)]

    values = np.concatenate(
        [values_a, values_b]
    )

    vmin = float(np.nanpercentile(values, 2))
    vmax = float(np.nanpercentile(values, 98))

    if vmax <= vmin:
        vmax = vmin + 1.0

    return vmin, vmax



def _query_raster_value(path, lon, lat):
    """Obtiene el valor del píxel original del GeoTIFF en una coordenada WGS84."""
    with rasterio.open(path) as src:
        if src.crs is None:
            return None, None, None

        # Convertimos la coordenada del clic desde WGS84 al CRS original del raster.
        xs, ys = rio_transform(
            "EPSG:4326",
            src.crs,
            [lon],
            [lat]
        )
        x, y = xs[0], ys[0]

        # Índice de la celda que contiene el punto.
        row, col = src.index(x, y)

        if row < 0 or row >= src.height or col < 0 or col >= src.width:
            return None, None, None

        value = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
        nodata = src.nodata

        if nodata is not None and np.isclose(value, nodata, equal_nan=True):
            return None, row, col

        if not np.isfinite(value):
            return None, row, col

        # Centro exacto del píxel consultado, en WGS84.
        center_x, center_y = rasterio.transform.xy(
            src.transform,
            row,
            col,
            offset="center"
        )
        center_lon, center_lat = rio_transform(
            src.crs,
            "EPSG:4326",
            [center_x],
            [center_y]
        )

        return float(value), row, col


def _show_pixel_query(click_data):
    """Muestra los valores GWRC y GWRCK del píxel seleccionado."""
    if not click_data or "lat" not in click_data or "lng" not in click_data:
        return

    lat = float(click_data["lat"])
    lon = float(click_data["lng"])

    gwrc_value, gwrc_row, gwrc_col = _query_raster_value(
        GWRC_FILE,
        lon,
        lat
    )
    gwrck_value, gwrck_row, gwrck_col = _query_raster_value(
        GWRCK_FILE,
        lon,
        lat
    )

    # Usamos el centro del píxel GWRC cuando está disponible.
    pixel_lat = lat
    pixel_lon = lon
    if gwrc_value is not None:
        with rasterio.open(GWRC_FILE) as src:
            xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
            row, col = src.index(xs[0], ys[0])
            cx, cy = rasterio.transform.xy(src.transform, row, col, offset="center")
            plon, plat = rio_transform(src.crs, "EPSG:4326", [cx], [cy])
            pixel_lon = float(plon[0])
            pixel_lat = float(plat[0])

    st.markdown("### SOC Value of the Selected Pixel")
    st.caption(
        f"Pixel center coordinates: {pixel_lat:.6f}°, {pixel_lon:.6f}°"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "GWRC",
            f"{gwrc_value:.2f} Mg ha⁻¹" if gwrc_value is not None else "No data"
        )
    with c2:
        st.metric(
            "GWRCK",
            f"{gwrck_value:.2f} Mg ha⁻¹" if gwrck_value is not None else "No data"
        )

    if gwrc_row is not None and gwrc_col is not None:
        st.caption(
            f"GWRC raster cell: row {gwrc_row + 1}, column {gwrc_col + 1}"
        )


def create_soc_map(gwrc_path, gwrck_path):
    """Construye el visor final con únicamente GWRC y GWRCK."""

    gwrc_data, gwrc_bounds = _read_raster_for_webmap(
        gwrc_path
    )

    gwrck_data, gwrck_bounds = _read_raster_for_webmap(
        gwrck_path
    )

    # Se utiliza una escala común para que ambas capas sean comparables.
    vmin, vmax = _common_scale(
        gwrc_data,
        gwrck_data
    )

    # El área de visualización se obtiene de la extensión combinada.
    all_bounds = [
        gwrc_bounds,
        gwrck_bounds
    ]

    south = min(
        bounds[0][0]
        for bounds in all_bounds
    )
    west = min(
        bounds[0][1]
        for bounds in all_bounds
    )
    north = max(
        bounds[1][0]
        for bounds in all_bounds
    )
    east = max(
        bounds[1][1]
        for bounds in all_bounds
    )

    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        control_scale=True,
        tiles=None
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
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

    gwrc_rgba = _continuous_rgba(
        gwrc_data,
        vmin,
        vmax
    )

    gwrck_rgba = _continuous_rgba(
        gwrck_data,
        vmin,
        vmax
    )

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
            text-shadow: 0 0 3px white, 0 0 3px white;
            font-weight: 600;
        }
    </style>
    """

    m.get_root().html.add_child(
        folium.Element(halo_css)
    )

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    return m


# -----------------------------------------------------------------------------
# INTERFAZ FINAL DEL VISOR
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# COMPROBACIÓN DE LOS RESULTADOS PRECALCULADOS
# -----------------------------------------------------------------------------

missing_files = []

if not GWRC_FILE.exists():
    missing_files.append(GWRC_FILE.name)

if not GWRCK_FILE.exists():
    missing_files.append(GWRCK_FILE.name)

if missing_files:
    st.error(
        "The following result files required by the viewer were not found: "
        + ", ".join(missing_files)
    )
    st.info(
        "Place the final GeoTIFF files inside the project's 'data' folder before deploying the application. "
        "End users do not need to upload any files."
    )
    st.stop()

# Single-screen landing layout: information on the left and the interactive
# spatial viewer on the right.
left_col, right_col = st.columns([0.38, 0.62], gap="large")

with left_col:
    st.markdown(
        '<div class="hero-title">Soil Organic Carbon Content and Spatial Distribution in the Amoju River Valley</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-location">Amoju River Valley, Jaen, Peru</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="context-card">
        The Amoju River Valley in northwestern Peru is an important agricultural area where
        rice cultivation represents a key productive resource. Soil organic carbon is an
        important component of soil functioning because its spatial distribution is related
        to soil quality, nutrient dynamics, and the capacity of agricultural soils to retain
        and cycle carbon.
        <br><br>
        This interactive map presents the spatial distribution of soil organic carbon estimated
        using the <b>GWRC</b> and <b>GWRCK</b> models. The resulting spatial information can help
        identify patterns and areas with contrasting soil carbon content across the agricultural
        landscape and support the planning of soil conservation and sustainable soil management
        strategies for rice production in the Amoju River Valley.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-card"><b>How to use the viewer</b></div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. Use the **Layer control** on the map to switch between GWRC and GWRCK.
        2. Zoom to the area of interest.
        3. **Click on any pixel** to retrieve its SOC value.
        4. The value is read from the original 30 × 30 m GeoTIFF.
        """
    )

    st.markdown('<div class="section-card"><b>Spatial models</b></div>', unsafe_allow_html=True)
    st.markdown(
        """
        **GWRC** — Geographically Weighted Regression with local ridge correction.  
        **GWRCK** — GWRC combined with kriged GWRC residuals.
        """
    )

    query_placeholder = st.empty()

with right_col:
    st.markdown('<div class="map-label">Interactive SOC spatial distribution</div>', unsafe_allow_html=True)
    st.caption(
        "Click on the map to retrieve the SOC content of the corresponding 30 × 30 m pixel. "
        "The two model layers use a common scale for comparison."
    )

    with st.spinner("Loading SOC map..."):
        soc_map = create_soc_map(
            GWRC_FILE,
            GWRCK_FILE
        )

    map_data = st_folium(
        soc_map,
        width=None,
        height=MAP_HEIGHT,
        returned_objects=["last_clicked"]
    )

click_data = map_data.get("last_clicked") if map_data else None

# The query result is placed in the left column so the map and information remain
# visible together on the landing screen.
with query_placeholder.container():
    if click_data:
        lat = float(click_data["lat"])
        lon = float(click_data["lng"])

        gwrc_value, gwrc_row, gwrc_col = _query_raster_value(
            GWRC_FILE,
            lon,
            lat
        )
        gwrck_value, _, _ = _query_raster_value(
            GWRCK_FILE,
            lon,
            lat
        )

        pixel_lat = lat
        pixel_lon = lon
        if gwrc_value is not None:
            with rasterio.open(GWRC_FILE) as src:
                xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
                row, col = src.index(xs[0], ys[0])
                cx, cy = rasterio.transform.xy(src.transform, row, col, offset="center")
                plon, plat = rio_transform(src.crs, "EPSG:4326", [cx], [cy])
                pixel_lon = float(plon[0])
                pixel_lat = float(plat[0])

        st.markdown('<div class="query-card">', unsafe_allow_html=True)
        st.markdown('<div class="query-title">SOC Value of the Selected Pixel</div>', unsafe_allow_html=True)
        st.caption(f"Pixel center: {pixel_lat:.6f}°, {pixel_lon:.6f}°")

        q1, q2 = st.columns(2)
        with q1:
            st.metric(
                "GWRC",
                f"{gwrc_value:.2f} Mg ha⁻¹" if gwrc_value is not None else "No data"
            )
        with q2:
            st.metric(
                "GWRCK",
                f"{gwrck_value:.2f} Mg ha⁻¹" if gwrck_value is not None else "No data"
            )

        if gwrc_row is not None and gwrc_col is not None:
            st.caption(f"GWRC raster cell: row {gwrc_row + 1}, column {gwrc_col + 1}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info(
            "Click on a pixel in the map to display its GWRC and GWRCK SOC values."
        )

# Additional information remains below the main two-column landing section.
st.divider()
st.markdown(
    """
    **Data interpretation**  
    SOC values are expressed as **Mg ha⁻¹**. Each pixel represents a 30 × 30 m spatial unit.
    The displayed model results were previously processed and are provided directly through
    this viewer; no model recalculation is performed by the application.
    """
)

st.markdown(
    """
    <div class="footer-note">
        Soil Organic Carbon Viewer · GWRC and GWRCK spatial models · Amoju River Valley, Jaen, Peru
    </div>
    """,
    unsafe_allow_html=True
)
