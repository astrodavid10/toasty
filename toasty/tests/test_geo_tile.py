# -*- mode: python; coding: utf-8 -*-
# Copyright 2026 the AAS WorldWide Telescope project
# Licensed under the MIT License.

"""Tests for localized planetary study and DEM tiling (toasty.geo_tile).

The geometry checks re-implement how the WWT engine places a TAN study in
planet mode (``Imageset._computeMatrix`` + ``TangentTile.geoTo3dTan``) and
assert that every sampled output pixel lands where GDAL says that pixel's
ground position is.
"""

import json
import math
import os

import numpy as np
import pytest

gdal = pytest.importorskip("osgeo.gdal")
osr = pytest.importorskip("osgeo.osr")

from ..geo_tile import PlanetDemTiler, PlanetStudyTiler, SIDECAR_NAME, study_imagesets  # noqa: E402
from ..study import StudyTiling  # noqa: E402

MOON_R = 1737400.0
MOON_GEOG = f"+proj=longlat +R={MOON_R} +no_defs"


# --- the engine's TAN placement, ported from @wwtelescope/engine -------------


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def engine_latlon(center_x, center_y, rotation, base_deg, offset_x, offset_y, p2n, gx, gy):
    """Selenographic (lat, east lon) WWT draws at global edge pixel (gx, gy)."""
    d = math.pi / 180
    m = _rx(rotation * d) @ _rz(center_y * d) @ _ry((360 - center_x) * d)  # row vectors: v @ m
    s = base_deg / p2n
    tlng = -base_deg / 2 + offset_x + gx * s
    tlat = base_deg / 2 + offset_y - gy * s
    half = base_deg / 2
    f = math.tan(half * d) / half
    v = np.array([1.0, tlat * f, -tlng * f]) @ m
    lat = math.degrees(math.asin(v[1] / np.linalg.norm(v)))
    lon = (math.degrees(math.atan2(v[2], v[0])) - 180 + 540) % 360 - 180
    return lat, lon


def test_engine_port_matches_engine():
    # Reference values measured with @wwtelescope/engine 7.40 in Node:
    # CenterX = lon + 180, tile +lat is north, tile +lng is west.
    lat, lon = engine_latlon(220.0, -85.0, 0, 2.0, 0.0, 0.0, 256, 128, 128)
    assert abs(lat + 85.0) < 1e-9 and abs(lon - 40.0) < 1e-9
    lat, lon = engine_latlon(220.0, -85.0, 0, 2.0, 0.0, 0.0, 256, 128 + 64, 128)  # +0.5° tile lng
    assert abs(lat + 84.975) < 1e-3 and abs(lon - 34.282) < 1e-3


# --- fixtures ----------------------------------------------------------------


def make_geographic_raster(path, lat0, lon0, half_lat, half_lon, size, func, dtype=gdal.GDT_Float32):
    """Equirectangular raster on the Moon sphere with values func(lat, lon)."""
    w, h = size
    ds = gdal.GetDriverByName("GTiff").Create(str(path), w, h, 1, dtype)
    dlon = 2 * half_lon / w
    dlat = 2 * half_lat / h
    ds.SetGeoTransform((lon0 - half_lon, dlon, 0, lat0 + half_lat, 0, -dlat))
    srs = osr.SpatialReference()
    srs.ImportFromProj4(MOON_GEOG)
    ds.SetProjection(srs.ExportToWkt())
    lons = lon0 - half_lon + (np.arange(w) + 0.5) * dlon
    lats = lat0 + half_lat - (np.arange(h) + 0.5) * dlat
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    ds.GetRasterBand(1).WriteArray(func(la, lo))
    ds.FlushCache()
    ds = None
    return str(path)


def gdal_latlon(grid, xu, yu):
    """(lat, lon) of an edge-based pixel position in the *unflipped* study grid."""
    gt = grid["geotransform"]
    px = gt[0] + xu * gt[1]
    py = gt[3] + yu * gt[5]
    src = osr.SpatialReference()
    src.ImportFromWkt(grid["wkt"])
    dst = osr.SpatialReference()
    dst.ImportFromProj4(MOON_GEOG)
    for s in (src, dst):
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lon, lat, _ = osr.CoordinateTransformation(src, dst).TransformPoint(px, py)
    return lat, lon


def angular_sep(a, b):
    la1, lo1 = map(math.radians, a)
    la2, lo2 = map(math.radians, b)
    c = math.sin(la1) * math.sin(la2) + math.cos(la1) * math.cos(la2) * math.cos(lo1 - lo2)
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


SITES = [
    (0.0, 10.0, 0.3, 0.3),  # equator
    (26.13, 3.63, 0.2, 0.25),  # Apollo 15
    (-85.0, 40.0, 0.25, 2.9),  # near the south pole: lat/lon grids are badly distorted here
]


@pytest.mark.parametrize("lat0,lon0,hlat,hlon", SITES)
def test_study_pixels_land_where_gdal_says(tmp_path, lat0, lon0, hlat, hlon):
    src = make_geographic_raster(tmp_path / "img.tif", lat0, lon0, hlat, hlon, (300, 240), lambda la, lo: la + lo)
    tiler = PlanetStudyTiler(src)
    params = tiler.study_parameters()
    grid = params["grid"]
    tiling = StudyTiling(tiler.width, tiler.height)
    p2n, gx0, gy0 = tiling._p2n, tiling._img_gx0, tiling._img_gy0
    km_per_deg = MOON_R * math.pi / 180 / 1000

    for fx, fy in [(0.5, 0.5), (0.05, 0.1), (0.95, 0.9), (0.1, 0.9), (0.9, 0.05)]:
        xf = fx * tiler.width  # flipped (stored) image coordinate
        yf = fy * tiler.height
        engine = engine_latlon(params["center_x"], params["center_y"], 0, params["base_degrees_per_tile"],
                               params["offset_x"], params["offset_y"], p2n, gx0 + xf, gy0 + yf)
        truth = gdal_latlon(grid, tiler.width - xf, yf)
        err_m = angular_sep(engine, truth) * km_per_deg * 1000
        assert err_m < 0.05 * tiler.resolution_m, (fx, fy, engine, truth, err_m)


def test_study_east_is_on_the_left_in_tile_space(tmp_path):
    # Stored images are flipped: tile X increases westward in planet mode.
    src = make_geographic_raster(tmp_path / "img.tif", 5.0, 20.0, 0.2, 0.2, (200, 200), lambda la, lo: lo)
    tiler = PlanetStudyTiler(src)
    p = tiler.study_parameters()
    tiling = StudyTiling(tiler.width, tiler.height)
    args = (p["center_x"], p["center_y"], 0, p["base_degrees_per_tile"], p["offset_x"], p["offset_y"], tiling._p2n)
    _, lon_left = engine_latlon(*args, tiling._img_gx0 + 10, tiling._img_gy0 + 100)
    _, lon_right = engine_latlon(*args, tiling._img_gx0 + tiler.width - 10, tiling._img_gy0 + 100)
    assert lon_left > lon_right


def test_dem_is_registered_to_the_study(tmp_path):
    lat0, lon0 = -12.0, 33.0

    def height(la, lo):
        return 800.0 * np.sin(np.radians(la - lat0) * 400) + 500.0 * np.cos(np.radians(lo - lon0) * 300)

    # Imagery and DEM on different grids: different resolution and extent.
    img = make_geographic_raster(tmp_path / "img.tif", lat0, lon0, 0.15, 0.15, (900, 900), lambda la, lo: la * 0 + 1)
    dem = make_geographic_raster(tmp_path / "dem.tif", lat0 + 0.01, lon0 - 0.01, 0.2, 0.2, (500, 500), height)

    out = tmp_path / "out"
    PlanetStudyTiler(img).tile(str(out), name="test", placeholder_thumbnail=True)
    wtml = str(out / "index_rel.wtml")
    n = PlanetDemTiler(dem, wtml).tile()
    assert n > 0

    grid = json.load(open(out / SIDECAR_NAME))
    tiling = StudyTiling(grid["width"], grid["height"])
    levels = tiling._tile_levels
    assert (out / "DEM" / "0" / "0" / "0_0.dem").exists()

    from wwt_data_formats.folder import Folder

    imgset = study_imagesets(Folder.from_file(wtml))[0]
    assert imgset.elevation_model
    assert imgset.dem_url == "DEM/{L}/{Y}/{Y}_{X}.dem"
    assert abs(imgset.mean_radius - MOON_R) < 1

    # Check a full-resolution tile whose 17×17 vertices all fall inside the image.
    ty = (tiling._img_gy0 + grid["height"] // 2) // 256
    tx = (tiling._img_gx0 + grid["width"] // 2) // 256
    vals = np.fromfile(out / "DEM" / str(levels) / str(ty) / f"{ty}_{tx}.dem", dtype="<f4").reshape(17, 17)
    worst = 0.0
    for r in range(0, 17, 4):
        for c in range(0, 17, 4):
            gx, gy = tx * 256 + c * 16, ty * 256 + r * 16
            la, lo = engine_latlon(imgset.center_x, imgset.center_y, 0, imgset.base_degrees_per_tile,
                                   imgset.offset_x, imgset.offset_y, tiling._p2n, gx, gy)
            worst = max(worst, abs(vals[r, c] - height(la, lo)))
    # Two bilinear resamplings of a field with ~4.5 km wavelength at ~1 m/px.
    assert worst < 15.0, worst
