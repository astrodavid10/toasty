# -*- mode: python; coding: utf-8 -*-
# Copyright 2024-2026 the AAS WorldWide Telescope project
# Licensed under the MIT License.

"""Localized tile pyramids for planetary imagery, with optional elevation.

A small, high-resolution region of a planet or moon (an LROC NAC frame, a
HiRISE strip, a mosaic of a landing site, …) is tiled as a WWT *study*: a
TAN-projected pyramid with ``DataSetType=Planet`` that WWT drapes over the
body's global map in planet mode. A matching DEM can then be tiled onto the
same grid so a DEM-aware engine can displace the study's mesh.

How the geometry works
----------------------
WWT draws a TAN study on a plane tangent to the body at ``(CenterY,
CenterX - 180)``. Tile coordinates are linear on that plane, i.e. the image
must be in a gnomonic projection centred on the study. So the input raster
is warped (with GDAL) straight into ``+proj=gnom`` centred on its own
footprint, with square pixels. That is exact at any latitude, including the
poles, where a lat/lon grid would be badly distorted.

In planet mode, increasing tile X runs *west*, so the warped raster is
stored flipped left-to-right. ``BaseDegreesPerTile`` and ``OffsetX/Y`` are
chosen so the projection centre lands on tile coordinate (0, 0).

The study tiler records the grid it used in ``planet_study.json`` next to
the WTML. The DEM tiler warps the DEM onto exactly that grid, so imagery and
elevation are registered by construction, whatever their source
projections and resolutions.

DEM tiles are 17×17 little-endian float32 heights in metres above the
reference radius (``MeanRadius``), row-major; row 0 is the tile's top
(``tv = 0``) and column 0 its left edge (``tu = 0``), matching the texture.
They live at ``DEM/{L}/{Y}/{Y}_{X}.dem`` and the WTML gets ``DemUrl`` and
``ElevationModel="True"``.

GDAL is imported lazily (``conda install -c conda-forge gdal``) so the rest
of toasty works without it.
"""

__all__ = """
PlanetStudyTiler
PlanetDemTiler
SIDECAR_NAME
""".split()

import json
import math
import os

import numpy as np

from .pyramid import Pos
from .study import StudyTiling

SIDECAR_NAME = "planet_study.json"
DEM_GRID = 17


# ---------------------------------------------------------------------------
# GDAL helpers
# ---------------------------------------------------------------------------


def _require_gdal():
    try:
        from osgeo import gdal, osr

        gdal.UseExceptions()
        return gdal, osr
    except ImportError:
        raise ImportError(
            "GDAL is required for planetary tiling. Install it with:\n"
            "  conda install -c conda-forge gdal"
        )


def _open(path):
    gdal, osr = _require_gdal()
    ds = gdal.Open(path)
    if ds is None:
        raise IOError(f"GDAL cannot open: {path}")
    wkt = ds.GetProjection()
    if not wkt:
        raise ValueError(f"No projection information found in {path}")
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    return ds, srs


def _footprint_center(ds, srs):
    """(lat, lon) in degrees of the raster's central pixel, lon in [-180, 180)."""
    gdal, osr = _require_gdal()
    geo = srs.CloneGeogCS()
    for s in (srs, geo):
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(srs, geo)
    gt = ds.GetGeoTransform()
    cx = gt[0] + gt[1] * ds.RasterXSize / 2 + gt[2] * ds.RasterYSize / 2
    cy = gt[3] + gt[4] * ds.RasterXSize / 2 + gt[5] * ds.RasterYSize / 2
    lon, lat, _ = ct.TransformPoint(cx, cy)
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat, lon


def _gnomonic_wkt(lat0, lon0, radius_m):
    _gdal, osr = _require_gdal()
    srs = osr.SpatialReference()
    srs.ImportFromProj4(
        f"+proj=gnom +lat_0={lat0!r} +lon_0={lon0!r} +R={radius_m!r} +units=m +no_defs"
    )
    return srs.ExportToWkt()


def _bilinear(arr, rows, cols):
    """Bilinear samples of ``arr`` at continuous pixel-centre coordinates (clamped)."""
    H, W = arr.shape
    rows = np.clip(rows, 0, H - 1)
    cols = np.clip(cols, 0, W - 1)
    r0 = np.floor(rows).astype(int)
    c0 = np.floor(cols).astype(int)
    r1 = np.minimum(r0 + 1, H - 1)
    c1 = np.minimum(c0 + 1, W - 1)
    fr = rows - r0
    fc = cols - c0
    return (
        arr[r0, c0] * (1 - fr) * (1 - fc)
        + arr[r0, c1] * (1 - fr) * fc
        + arr[r1, c0] * fr * (1 - fc)
        + arr[r1, c1] * fr * fc
    )


def _halve(arr):
    """2× nan-aware mean downsample (odd edges are padded by repetition)."""
    H, W = arr.shape
    if H % 2:
        arr = np.vstack([arr, arr[-1:]])
    if W % 2:
        arr = np.hstack([arr, arr[:, -1:]])
    blocks = arr.reshape(arr.shape[0] // 2, 2, arr.shape[1] // 2, 2)
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=(1, 3))


# ---------------------------------------------------------------------------
# Imagery
# ---------------------------------------------------------------------------


class PlanetStudyTiler:
    """Tile a georeferenced planetary raster as a localized WWT study.

    Parameters
    ----------
    path : str
        Any GDAL-readable raster with a projection (GeoTIFF, PDS ``.IMG``,
        ISIS cube, …). One band (greyscale) or three/four (RGB/RGBA).
    radius_m : float, optional
        Body radius in metres. Defaults to the semi-major axis of the
        raster's CRS (1737400 for the Moon's IAU sphere).
    resolution_m : float, optional
        Output pixel size in metres. Defaults to GDAL's estimate of the
        source's native resolution.

    Example
    -------
    ::

        tiler = PlanetStudyTiler("M1234_NAC_ortho.tif")
        tiler.tile("out/", name="Apollo 15 landing site")
        PlanetDemTiler("M1234_NAC_dtm.tif", "out/index_rel.wtml").tile()
    """

    def __init__(self, path, radius_m=None, resolution_m=None):
        gdal, _osr = _require_gdal()
        self.path = path
        src, srs = _open(path)
        self.radius_m = float(radius_m or srs.GetSemiMajor())
        self.center_lat, self.center_lon = _footprint_center(src, srs)
        self.wkt = _gnomonic_wkt(self.center_lat, self.center_lon, self.radius_m)

        if resolution_m is None:
            vrt = gdal.AutoCreateWarpedVRT(src, None, self.wkt, gdal.GRA_Bilinear)
            vgt = vrt.GetGeoTransform()
            resolution_m = min(abs(vgt[1]), abs(vgt[5]))
            vrt = None
        self.resolution_m = float(resolution_m)

        nodata = src.GetRasterBand(1).GetNoDataValue()
        self._ds = gdal.Warp(
            "",
            src,
            format="MEM",
            dstSRS=self.wkt,
            xRes=self.resolution_m,
            yRes=self.resolution_m,
            resampleAlg=gdal.GRA_Bilinear,
            srcNodata=nodata,
            dstAlpha=True,
        )
        self.width = self._ds.RasterXSize
        self.height = self._ds.RasterYSize
        self.geotransform = self._ds.GetGeoTransform()

    def _read_rgba(self):
        """The warped raster as flipped (east-left) RGBA uint8, transparent off-image."""
        ds = self._ds
        n = ds.RasterCount  # last band is the alpha GDAL added
        alpha = ds.GetRasterBand(n).ReadAsArray() > 0
        bands = [ds.GetRasterBand(b + 1).ReadAsArray() for b in range(n - 1)]
        data = np.stack(bands, axis=-1).astype(np.float32)
        if data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=-1)
        data = data[..., :3]

        if self._ds.GetRasterBand(1).DataType != gdal_byte():
            valid = data[alpha]
            lo, hi = np.percentile(valid, [0.5, 99.5]) if valid.size else (0.0, 1.0)
            scale = 255.0 / (hi - lo) if hi > lo else 0.0
            data = np.clip((data - lo) * scale, 0, 255)

        rgba = np.empty(data.shape[:2] + (4,), dtype=np.uint8)
        rgba[..., :3] = data.astype(np.uint8)
        rgba[..., 3] = np.where(alpha, 255, 0)
        return np.ascontiguousarray(rgba[:, ::-1])

    def study_parameters(self):
        """WTML positioning for the study, and the grid record for the sidecar."""
        tiling = StudyTiling(self.width, self.height)
        p2n = tiling._p2n
        gx0, gy0 = tiling._img_gx0, tiling._img_gy0
        gt = self.geotransform
        # Projection centre (plane origin) in edge-based pixel coordinates of
        # the unflipped raster, then of the flipped one, then global.
        cx = -gt[0] / gt[1]
        cy = -gt[3] / gt[5]
        gx = gx0 + (self.width - cx)
        gy = gy0 + cy
        # The base tile spans p2n pixels; on the tangent plane that is
        # 2·tan(B/2) body radii.
        base_deg = 2.0 * math.degrees(math.atan(p2n * self.resolution_m / (2.0 * self.radius_m)))
        deg_per_px = base_deg / p2n
        return {
            "center_x": (self.center_lon + 180.0) % 360.0,
            "center_y": self.center_lat,
            "base_degrees_per_tile": base_deg,
            "offset_x": base_deg / 2.0 - gx * deg_per_px,
            "offset_y": gy * deg_per_px - base_deg / 2.0,
            "grid": {
                "wkt": self.wkt,
                "geotransform": list(gt),
                "width": self.width,
                "height": self.height,
                "flipped_x": True,
                "radius_m": self.radius_m,
                "center_lat": self.center_lat,
                "center_lon": self.center_lon,
            },
        }

    def tile(self, output_dir, name="Toasty", placeholder_thumbnail=False, cli_progress=False):
        """Write the pyramid, ``index_rel.wtml`` and ``planet_study.json``.

        Returns the :class:`toasty.builder.Builder`.
        """
        from wwt_data_formats.enums import DataSetType, ProjectionType

        from .builder import Builder
        from .image import Image
        from .pyramid import PyramidIO

        img = Image.from_array(self._read_rgba())
        os.makedirs(output_dir, exist_ok=True)
        builder = Builder(PyramidIO(output_dir, default_format="png"))
        tiling = builder.prepare_study_tiling(img)

        params = self.study_parameters()
        imgset = builder.imgset
        imgset.data_set_type = DataSetType.PLANET
        imgset.projection = ProjectionType.TAN
        imgset.bottoms_up = False
        imgset.width_factor = 2
        imgset.rotation_deg = 0.0
        imgset.center_x = params["center_x"]
        imgset.center_y = params["center_y"]
        imgset.base_degrees_per_tile = params["base_degrees_per_tile"]
        imgset.offset_x = params["offset_x"]
        imgset.offset_y = params["offset_y"]
        imgset.mean_radius = self.radius_m

        place = builder.place
        place.data_set_type = DataSetType.PLANET
        place.latitude = self.center_lat
        place.longitude = self.center_lon
        span_deg = math.degrees(max(self.width, self.height) * self.resolution_m / self.radius_m)
        place.zoom_level = span_deg * 6 * 1.2

        if placeholder_thumbnail:
            builder.make_placeholder_thumbnail()
        else:
            builder.make_thumbnail_from_other(img)
        builder.execute_study_tiling(img, tiling, cli_progress=cli_progress)
        builder.set_name(name)
        builder.write_index_rel_wtml()

        with open(os.path.join(output_dir, SIDECAR_NAME), "w") as f:
            json.dump(params["grid"], f, indent=1)
        return builder


def study_imagesets(folder):
    """Every ImageSet in a folder, directly or via its Places."""
    from wwt_data_formats.imageset import ImageSet
    from wwt_data_formats.place import Place

    out = []
    for child in folder.children:
        if isinstance(child, ImageSet):
            out.append(child)
        elif isinstance(child, Place):
            for attr in ("foreground_image_set", "background_image_set", "image_set"):
                iset = getattr(child, attr, None)
                if iset is not None:
                    out.append(iset)
    return out


def gdal_byte():
    gdal, _osr = _require_gdal()
    return gdal.GDT_Byte


# ---------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------


class PlanetDemTiler:
    """Tile a DEM onto the grid of an existing planetary study.

    Parameters
    ----------
    dem_path : str
        Any GDAL-readable elevation raster in metres.
    wtml_path : str
        The ``index_rel.wtml`` written by :class:`PlanetStudyTiler`; its
        ``planet_study.json`` sidecar must sit next to it.
    offset_m : float
        Subtracted from every sample. Use it for DEMs that store planetary
        *radius* rather than height (e.g. ``1737400`` for some lunar
        products), or to change the reference.
    fill_search_px : int
        How far (pixels) GDAL may reach to fill holes in the DEM. Anything
        still empty is set to the DEM's mean height.
    """

    def __init__(self, dem_path, wtml_path, offset_m=0.0, fill_search_px=100):
        gdal, _osr = _require_gdal()
        self.dem_path = dem_path
        self.wtml_path = wtml_path
        self.out_dir = os.path.dirname(os.path.abspath(wtml_path))
        sidecar = os.path.join(self.out_dir, SIDECAR_NAME)
        if not os.path.exists(sidecar):
            raise FileNotFoundError(
                f"{sidecar} not found; tile the imagery with PlanetStudyTiler (tile-planet-study) first"
            )
        with open(sidecar) as f:
            self.grid = json.load(f)

        g = self.grid
        gt = g["geotransform"]
        w, h = g["width"], g["height"]
        bounds = (gt[0], gt[3] + gt[5] * h, gt[0] + gt[1] * w, gt[3])  # minX, minY, maxX, maxY

        src, _srs = _open(dem_path)
        src_nodata = src.GetRasterBand(1).GetNoDataValue()
        warped = gdal.Warp(
            "",
            src,
            format="MEM",
            dstSRS=g["wkt"],
            outputBounds=bounds,
            width=w,
            height=h,
            resampleAlg=gdal.GRA_Bilinear,
            srcNodata=src_nodata,
            dstNodata=np.nan,
            outputType=gdal.GDT_Float32,
        )
        band = warped.GetRasterBand(1)
        if fill_search_px > 0:
            gdal.FillNodata(band, None, fill_search_px, 0)
        dem = band.ReadAsArray().astype(np.float32) - np.float32(offset_m)
        if np.isnan(dem).all():
            raise ValueError(f"{dem_path} does not overlap the study area")
        dem[np.isnan(dem)] = np.nanmean(dem)
        self.dem = np.ascontiguousarray(dem[:, ::-1]) if g.get("flipped_x", True) else dem

    def _tile_values(self, level_dem, level_scale, gx0, gy0, pos):
        """17×17 heights for tile ``pos``, sampled from a DEM averaged ``level_scale``×."""
        k = np.arange(DEM_GRID) * (256 // (DEM_GRID - 1))
        # Edge-based global pixel coordinates at this level → full-res image
        # coordinates → pixel-centre coordinates in the level's DEM.
        gx = (pos.x * 256 + k) * level_scale - gx0
        gy = (pos.y * 256 + k) * level_scale - gy0
        cols = gx / level_scale - 0.5
        rows = gy / level_scale - 0.5
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        return _bilinear(level_dem, rr, cc).astype("<f4")

    def tile(self, cli_progress=False):
        """Write ``DEM/{L}/{Y}/{Y}_{X}.dem`` for every imagery tile and update the WTML."""
        from wwt_data_formats import write_xml_doc
        from wwt_data_formats.folder import Folder

        g = self.grid
        tiling = StudyTiling(g["width"], g["height"])
        levels = tiling._tile_levels
        gx0, gy0 = tiling._img_gx0, tiling._img_gy0

        level_dem = self.dem
        count = 0
        for level in range(levels, -1, -1):
            scale = 2 ** (levels - level)
            # Image footprint in this level's tile grid.
            tx0 = (gx0 // scale) // 256
            ty0 = (gy0 // scale) // 256
            tx1 = ((gx0 + g["width"] - 1) // scale) // 256
            ty1 = ((gy0 + g["height"] - 1) // scale) // 256
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    pos = Pos(n=level, x=tx, y=ty)
                    values = self._tile_values(level_dem, scale, gx0, gy0, pos)
                    tile_dir = os.path.join(self.out_dir, "DEM", str(level), str(ty))
                    os.makedirs(tile_dir, exist_ok=True)
                    with open(os.path.join(tile_dir, f"{ty}_{tx}.dem"), "wb") as f:
                        f.write(values.tobytes())
                    count += 1
            if level > 0:
                level_dem = _halve(level_dem)
            if cli_progress:
                print(f"DEM level {level}: done")

        fld = Folder.from_file(self.wtml_path)
        imgsets = study_imagesets(fld)
        if not imgsets:
            raise ValueError(f"No ImageSet found in {self.wtml_path}")
        for imgset in imgsets:
            imgset.dem_url = "DEM/{L}/{Y}/{Y}_{X}.dem"
            imgset.elevation_model = True
            imgset.mean_radius = g["radius_m"]
        with open(self.wtml_path, "wb") as f:
            write_xml_doc(fld.to_xml(), dest_stream=f, dest_wants_bytes=True)
        return count
