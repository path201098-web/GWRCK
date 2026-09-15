import os
from pathlib import Path

import numpy as np
import streamlit as st
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

import folium
from folium.raster_layers import ImageOverlay
from branca.colormap import LinearColormap
from streamlit_folium import st_folium


st.set_page_config(
    page_title="Visor SOCS | Valle del río Amojú",
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
        .main-title {
            text-align: center;
            margin-bottom: 0.2rem;
        }

        .main-subtitle {
            text-align: center;
            color: #5f6368;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .viewer-note {
            background: rgba(240, 244, 248, 0.75);
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(120, 120, 120, 0.18);
        }

        .footer-note {
            text-align: center;
            color: #6b7280;
            font-size: 0.85rem;
            margin-top: 1.5rem;
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
        caption="SOCS (Mg ha⁻¹)"
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

st.markdown(
    '<h1 class="main-title">Mapa Digital de Carbono Orgánico del Suelo</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">Valle del río Amojú, Jaén, Perú</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="viewer-note">
        <b>Visor de SOCS</b><br>
        Visualización espacial de las estimaciones de carbono orgánico del suelo
        mediante los modelos GWRC y GWRCK. Los datos y resultados del modelo
        fueron previamente procesados y cargados por el desarrollador.
    </div>
    """,
    unsafe_allow_html=True
)


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
        "No se encontraron los archivos de resultados necesarios para el visor: "
        + ", ".join(missing_files)
    )

    st.info(
        "Coloca los GeoTIFF finales dentro de la carpeta 'data' del proyecto "
        "antes de desplegar la aplicación. El usuario final no tendrá que cargar "
        "ningún archivo."
    )

    st.stop()


st.subheader("Visualización espacial")

st.write(
    "Utiliza el control de capas ubicado en la esquina superior derecha "
    "del mapa para activar o desactivar GWRC y GWRCK."
)


with st.spinner("Cargando mapa de SOCS..."):
    soc_map = create_soc_map(
        GWRC_FILE,
        GWRCK_FILE
    )

st_folium(
    soc_map,
    width=None,
    height=MAP_HEIGHT,
    returned_objects=[]
)


st.markdown(
    """
    <div class="footer-note">
        SOCS expresado en Mg ha⁻¹ · Modelos espaciales GWRC y GWRCK
    </div>
    """,
    unsafe_allow_html=True
)
