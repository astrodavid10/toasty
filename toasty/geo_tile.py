# -*- mode: python; coding: utf-8 -*-
# Copyright 2024 the AAS WorldWide Telescope project
# Licensed under the MIT License.

"""Tiling of ortho-projected planetary GeoTIFF imagery and DEMs.

Requires GDAL (``conda install -c conda-forge gdal``). All GDAL imports are
lazy so that the rest of toasty can be imported without GDAL installed.
"""

__all__ = """
PlanetStudyTiler
PlanetDemTiler
""".split()

import math
import os
import struct

import numpy as np

from .pyramid import next_highest_power_of_2, Pos
from .study import StudyTiling


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
            "GDAL is required for planet tiling. Install it with:\n"
            "  conda install -c conda-forge gdal"
        )


def _open_warp_geographic(path):
    """Open *path* with GDAL and warp it to north-up geographic (lat/lon).

    Returns ``(warped_ds, geo_srs, planet_radius_m, nodata)`` where
    *warped_ds* is a GDAL in-memory dataset in the body's geographic CRS.
    """
    gdal, osr = _require_gdal()

    src_ds = gdal.Open(path)
    if src_ds is None:
        raise IOError(f"GDAL cannot open: {path}")

    proj_wkt = src_ds.GetProjection()
    if not proj_wkt:
        raise ValueError(f"No projection information found in {path}")

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(proj_wkt)

    # Strip to the geographic (lat/lon) CRS of the same body.
    geo_srs = src_srs.CloneGeogCS()
    geo_wkt = geo_srs.ExportToWkt()
    src_wkt = src_srs.ExportToWkt()

    # Mean radius from the ellipsoid definition (semi-major axis).
    planet_radius_m = geo_srs.GetSemiMajor()

    # Use AutoCreateWarpedVRT to compute the correct output extent/resolution.
    vrt = gdal.AutoCreateWarpedVRT(src_ds, src_wkt, geo_wkt, gdal.GRA_Bilinear)
    if vrt is None:
        raise RuntimeError(f"Failed to create warped VRT for {path}")

    out_w = vrt.RasterXSize
    out_h = vrt.RasterYSize
    out_gt = vrt.GetGeoTransform()
    n_bands = vrt.RasterCount
    dtype = vrt.GetRasterBand(1).DataType

    nodata = src_ds.GetRasterBand(1).GetNoDataValue()

    mem_drv = gdal.GetDriverByName("MEM")
    out_ds = mem_drv.Create("", out_w, out_h, n_bands, dtype)
    out_ds.SetGeoTransform(out_gt)
    out_ds.SetProjection(geo_wkt)
    if nodata is not None:
        for b in range(1, n_bands + 1):
            out_ds.GetRasterBand(b).SetNoDataValue(nodata)

    gdal.ReprojectImage(src_ds, out_ds, src_wkt, geo_wkt, gdal.GRA_Bilinear)

    return out_ds, geo_srs, planet_radius_m, nodata


def _geo_params(ds):
    """Return ``(center_lon, center_lat, pixel_scale_deg)`` for a geographic GDAL dataset."""
    gt = ds.GetGeoTransform()
    w = ds.RasterXSize
    h = ds.RasterYSize

    # gt[0]=UL lon, gt[1]=lon/pixel, gt[3]=UL lat, gt[5]=lat/pixel (<0 for north-up)
    center_lon = gt[0] + w * gt[1] / 2.0
    center_lat = gt[3] + h * gt[5] / 2.0
    pixel_scale_deg = (abs(gt[1]) + abs(gt[5])) / 2.0

    return center_lon, center_lat, pixel_scale_deg


def _bilinear_sample(arr, rows, cols, nodata_value, fill=0.0):
    """Bilinear interpolation of *arr* at floating-point (rows, cols).

    Out-of-bounds and nodata positions are replaced with *fill*.
    """
    H, W = arr.shape
    r0 = np.floor(rows).astype(int)
    c0 = np.floor(cols).astype(int)
    r1 = r0 + 1
    c1 = c0 + 1
    fr = rows - r0
    fc = cols - c0

    r0c = np.clip(r0, 0, H - 1)
    r1c = np.clip(r1, 0, H - 1)
    c0c = np.clip(c0, 0, W - 1)
    c1c = np.clip(c1, 0, W - 1)

    v00 = arr[r0c, c0c].astype(np.float32)
    v01 = arr[r0c, c1c].astype(np.float32)
    v10 = arr[r1c, c0c].astype(np.float32)
    v11 = arr[r1c, c1c].astype(np.float32)

    result = (v00 * (1 - fc) * (1 - fr)
              + v01 * fc * (1 - fr)
              + v10 * (1 - fc) * fr
              + v11 * fc * fr)

    oob = (r0 < 0) | (r1 >= H) | (c0 < 0) | (c1 >= W)
    if nodata_value is not None:
        nd = np.float32(nodata_value)
        tol = max(1e-6 * abs(float(nd) + 1), 1e-20)
        nodata_mask = (
            (np.abs(v00 - nd) < tol)
            | (np.abs(v01 - nd) < tol)
            | (np.abs(v10 - nd) < tol)
            | (np.abs(v11 - nd) < tol)
        )
        oob |= nodata_mask

    result[oob] = fill
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Imagery tiling
# ---------------------------------------------------------------------------

class PlanetStudyTiler:
    """Tile an ortho-projected planetary GeoTIFF as a WWT "study" (TAN projection).

    Parameters
    ----------
    geotiff_path : str
        Path to the ortho-projected GeoTIFF (any GDAL-supported format).

    Usage
    -----
    ::

        tiler = PlanetStudyTiler("image.tif")
        builder = tiler.tile("/output/dir", name="My Mars Imagery", cli_progress=True)

    The output directory will contain a tile pyramid and an ``index_rel.wtml``
    that positions the imagery on the correct planet surface in WWT.
    """

    def __init__(self, geotiff_path):
        self.path = geotiff_path
        warped_ds, geo_srs, planet_radius_m, nodata = _open_warp_geographic(geotiff_path)

        self._ds = warped_ds
        self._geo_srs = geo_srs
        self._nodata = nodata

        self.planet_radius_m = planet_radius_m
        self.width = warped_ds.RasterXSize
        self.height = warped_ds.RasterYSize
        self.n_bands = warped_ds.RasterCount

        self.center_lon, self.center_lat, self.pixel_scale_deg = _geo_params(warped_ds)

    def _read_as_image(self):
        from .image import Image

        if self.n_bands == 1:
            data = self._ds.GetRasterBand(1).ReadAsArray()
            if self._nodata is not None:
                data[data == self._nodata] = 0
            # Convert grayscale to 3-channel RGB so PNG output works cleanly.
            data = np.stack([data, data, data], axis=-1)
        else:
            bands = [self._ds.GetRasterBand(b + 1).ReadAsArray() for b in range(self.n_bands)]
            data = np.stack(bands, axis=-1)
            if self._nodata is not None:
                mask = np.zeros(data.shape[:2], dtype=bool)
                for b in range(self.n_bands):
                    mask |= (bands[b] == self._nodata)
                data[mask] = 0

        if data.dtype != np.uint8:
            lo, hi = np.nanpercentile(data, [0.5, 99.5])
            if hi > lo:
                data = np.clip((data.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
            else:
                data = np.zeros_like(data, dtype=np.uint8)

        return Image.from_array(data)

    def tile(self, output_dir, name="Toasty", placeholder_thumbnail=False, cli_progress=False):
        """Tile the image and write the pyramid and WTML.

        Parameters
        ----------
        output_dir : str
            Root directory for tile output. Created if it does not exist.
        name : str
            Image name embedded in the WTML.
        placeholder_thumbnail : bool
            If True, skip generating a real thumbnail (saves memory for very large inputs).
        cli_progress : bool
            Show a progress bar.

        Returns
        -------
        builder : :class:`toasty.builder.Builder`
            The builder with the finished imageset ready for inspection.
        """
        from .builder import Builder
        from .pyramid import PyramidIO

        img = self._read_as_image()
        pio = PyramidIO(output_dir, default_format="png")
        builder = Builder(pio)

        tiling = builder.prepare_study_tiling(img)
        self._apply_planet_wcs(builder, tiling)

        if placeholder_thumbnail:
            builder.make_placeholder_thumbnail()
        else:
            builder.make_thumbnail_from_other(img)

        builder.execute_study_tiling(img, tiling, cli_progress=cli_progress)
        builder.set_name(name)
        builder.write_index_rel_wtml()

        return builder

    def _apply_planet_wcs(self, builder, tiling):
        """Set all ImageSet positioning fields for a planetary TAN study."""
        from wwt_data_formats.enums import DataSetType, ProjectionType

        imgset = builder.imgset
        p2n = next_highest_power_of_2(max(self.width, self.height))

        imgset.data_set_type = DataSetType.PLANET
        imgset.projection = ProjectionType.TAN
        imgset.bottoms_up = False
        imgset.width_factor = 2
        imgset.center_x = self.center_lon
        imgset.center_y = self.center_lat
        imgset.offset_x = 0.0
        imgset.offset_y = 0.0
        imgset.rotation_deg = 0.0
        # base_degrees_per_tile = angular extent of the full padded (p2n) pixel grid
        imgset.base_degrees_per_tile = self.pixel_scale_deg * p2n
        imgset.mean_radius = self.planet_radius_m

        # Set a reasonable zoom level on the accompanying Place.
        image_height_deg = self.pixel_scale_deg * self.height
        builder.place.zoom_level = image_height_deg * 1.7 * 6
        builder.place.data_set_type = DataSetType.PLANET


# ---------------------------------------------------------------------------
# DEM tiling
# ---------------------------------------------------------------------------

# WWT's TangentTile DEM tile format (as implemented in this project's
# tangent_tile.js): a 17×17 grid of float32 elevation values (metres), stored
# row-major where row 0 is the north edge of the tile (tv=0) and col 0 is the
# west edge (tu=0).
_DEM_GRID = 17
_DEM_SAMPLES = _DEM_GRID * _DEM_GRID  # 289


class PlanetDemTiler:
    """Tile a planetary DEM alongside a study pyramid.

    Reads any GDAL-readable raster (GeoTIFF, PDS .IMG, etc.) containing
    float32 elevation values, warps it to geographic lat/lon, then generates
    17×17 float32 binary DEM tiles that match the study tile scheme of the
    companion imagery pyramid.

    The output DEM tiles are placed in a ``DEM/`` sub-directory next to the
    study imagery, and the companion WTML is updated in-place to reference
    them.

    Parameters
    ----------
    dem_path : str
        Path to the DEM raster.
    imagery_wtml_path : str
        Path to the ``index_rel.wtml`` produced by :class:`PlanetStudyTiler`.
        The DEM URL and ``ElevationModel=True`` flag will be written into it.
    """

    def __init__(self, dem_path, imagery_wtml_path):
        self.dem_path = dem_path
        self.wtml_path = imagery_wtml_path

        warped_ds, geo_srs, planet_radius_m, nodata = _open_warp_geographic(dem_path)
        self._ds = warped_ds
        self._nodata = nodata
        self._gt = warped_ds.GetGeoTransform()
        self.dem_width = warped_ds.RasterXSize
        self.dem_height = warped_ds.RasterYSize

        # Load full DEM into memory (float32)
        raw = warped_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        if nodata is not None:
            raw[raw == np.float32(nodata)] = 0.0
        self._dem_array = raw

    def _sample_tile_dem(self, pos, p2n, img_gx0, img_gy0, tile_levels):
        """Return a 17×17 float32 array of elevations for tile *pos*.

        Samples are at 17 evenly-spaced positions across the tile in both
        directions (including both edge samples).
        """
        scale = 2 ** (tile_levels - pos.n)  # full-res pixels per level-L pixel

        # 17 sample positions in global (padded) pixel coordinates
        gx_samples = pos.x * 256 * scale + np.arange(_DEM_GRID) * 16 * scale
        gy_samples = pos.y * 256 * scale + np.arange(_DEM_GRID) * 16 * scale

        # Convert global pixels → image pixels
        ix = gx_samples - img_gx0
        iy = gy_samples - img_gy0

        # Convert image pixels → geographic coordinates
        gt = self._gt
        # gt[0]=UL lon, gt[1]=lon/px, gt[3]=UL lat, gt[5]=lat/px (<0)
        lons = gt[0] + (ix + 0.5) * gt[1]  # (17,)
        lats = gt[3] + (iy + 0.5) * gt[5]  # (17,)

        # Convert lat/lon → DEM raster pixel coordinates
        dem_cols = (lons - gt[0]) / gt[1] - 0.5  # (17,)
        dem_rows = (lats - gt[3]) / gt[5] - 0.5  # (17,)

        # Build 17×17 grids (row = lat, col = lon)
        dem_row_grid, dem_col_grid = np.meshgrid(dem_rows, dem_cols, indexing="ij")

        return _bilinear_sample(self._dem_array, dem_row_grid, dem_col_grid, None)

    def _write_dem_tile(self, dem_dir, pos, values):
        level_dir = os.path.join(dem_dir, str(pos.n), str(pos.y))
        os.makedirs(level_dir, exist_ok=True)
        fname = f"{pos.y}_{pos.x}.dem"
        data = values.astype("<f4").tobytes()  # little-endian float32
        with open(os.path.join(level_dir, fname), "wb") as f:
            f.write(data)

    def tile(self, cli_progress=False):
        """Generate all DEM tiles and update the companion WTML.

        DEM tiles are written into a ``DEM/`` sub-directory next to
        ``imagery_wtml_path``.
        """
        from wwt_data_formats.folder import Folder

        # --- Parse the companion WTML to get tiling parameters ---
        wtml_dir = os.path.dirname(os.path.abspath(self.wtml_path))
        dem_dir = os.path.join(wtml_dir, "DEM")

        fld = Folder.from_file(self.wtml_path)
        imgset = None
        for _idx, _kind, item in fld.immediate_imagesets():
            imgset = item
            break

        if imgset is None:
            # Try via Place
            for _idx, _kind, item in fld.immediate_places():
                if item.foreground_image_set is not None:
                    imgset = item.foreground_image_set
                    break

        if imgset is None:
            raise ValueError(f"No ImageSet found in {self.wtml_path}")

        tile_levels = imgset.tile_levels
        base_deg = imgset.base_degrees_per_tile  # = pixel_scale_deg * p2n
        center_lon = imgset.center_x
        center_lat = imgset.center_y

        # Reconstruct geometry
        # pixel_scale_deg = base_deg / p2n, and p2n = 2^tile_levels * 256
        p2n = (2 ** tile_levels) * 256
        pixel_scale_deg = base_deg / p2n

        # Reconstruct the image's width and height from the pixel offset fields.
        # offset_x/y are 0.0 (image center = coordinate center), so the padded
        # image is p2n x p2n and the actual image is whatever the tiler used.
        # We need img_gx0/img_gy0.  With offsets = 0, the image is centered:
        #   img_gx0 = (p2n - width) // 2
        # We recover width/height from tile_levels & the WTML base tile degrees.
        # Actually we only need img_gx0 which depends on image width.
        # Since offset_x == 0, the formula from wwt_data_formats says:
        #   offset_x = ((width+1)//2 - refpix_x) * scale_x = 0
        # => refpix_x = (width+1)//2
        # But we don't have width in the WTML. We use a fallback:
        # Scan the deepest tile directory to infer covered area.
        img_gx0, img_gy0, img_width, img_height = self._infer_image_geometry(
            wtml_dir, tile_levels
        )

        tiling = _StudyTilingProxy(tile_levels, img_gx0, img_gy0, img_width, img_height, p2n)

        # --- Generate DEM tiles ---
        positions = list(tiling.generate_populated_positions())

        try:
            from tqdm import tqdm
            iterator = tqdm(positions, desc="DEM tiles") if cli_progress else positions
        except ImportError:
            iterator = positions

        for pos, *_ in iterator:
            values = self._sample_tile_dem(pos, p2n, img_gx0, img_gy0, tile_levels)
            self._write_dem_tile(dem_dir, pos, values)

        # Also generate coarser-level tiles by averaging children
        for level in range(tile_levels - 1, -1, -1):
            self._cascade_dem_level(dem_dir, level, tile_levels, p2n, img_gx0, img_gy0)

        # --- Update the WTML ---
        dem_url = "DEM/{L}/{Y}/{Y}_{X}.dem"
        imgset.dem_url = dem_url
        imgset.elevation_model = True

        # Set the planet mean radius so the engine scales DEM values correctly.
        gdal, osr = _require_gdal()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self._ds.GetProjection())
        geo_srs = srs.CloneGeogCS()
        imgset.mean_radius = geo_srs.GetSemiMajor()

        from wwt_data_formats import write_xml_doc
        with open(self.wtml_path, "wb") as f:
            write_xml_doc(fld.to_xml(), dest_stream=f, dest_wants_bytes=True)

        print(f"DEM tiles written to: {dem_dir}")
        print(f"WTML updated: {self.wtml_path}")

    def _infer_image_geometry(self, wtml_dir, tile_levels):
        """Scan the deepest tile level to find the image bounds in global pixels."""
        level_dir = os.path.join(wtml_dir, str(tile_levels))
        if not os.path.isdir(level_dir):
            raise RuntimeError(
                f"Cannot find tile directory {level_dir}; run tile-planet-study first."
            )

        min_x = min_y = 10**9
        max_x = max_y = -1

        for ydir in os.listdir(level_dir):
            try:
                iy = int(ydir)
            except ValueError:
                continue
            tile_ydir = os.path.join(level_dir, ydir)
            for fname in os.listdir(tile_ydir):
                base = os.path.splitext(fname)[0]
                if "_" not in base:
                    continue
                try:
                    fy, fx = (int(v) for v in base.split("_"))
                except ValueError:
                    continue
                min_x = min(min_x, fx)
                max_x = max(max_x, fx)
                min_y = min(min_y, fy)
                max_y = max(max_y, fy)

        if max_x < 0:
            raise RuntimeError(f"No tiles found under {level_dir}")

        # Tile coverage in global pixels
        gx0 = min_x * 256
        gy0 = min_y * 256
        gx1 = (max_x + 1) * 256
        gy1 = (max_y + 1) * 256

        return gx0, gy0, gx1 - gx0, gy1 - gy0

    def _cascade_dem_level(self, dem_dir, level, tile_levels, p2n, img_gx0, img_gy0):
        """Generate level *level* DEM tiles by re-sampling the full-res DEM."""
        n_tiles = 2 ** level
        scale = 2 ** (tile_levels - level)  # full-res pixels per level-L pixel

        img_gx1 = img_gx0 + p2n  # padded image right edge
        img_gy1 = img_gy0 + p2n  # padded image bottom edge

        for ty in range(n_tiles):
            for tx in range(n_tiles):
                # Global pixel range of this tile (at full-resolution scale)
                gx0_tile = tx * 256 * scale
                gx1_tile = (tx + 1) * 256 * scale
                gy0_tile = ty * 256 * scale
                gy1_tile = (ty + 1) * 256 * scale

                # Skip if the tile doesn't overlap with any image data
                if gx1_tile <= img_gx0 or gx0_tile >= img_gx1:
                    continue
                if gy1_tile <= img_gy0 or gy0_tile >= img_gy1:
                    continue

                tile_dem_path = os.path.join(
                    dem_dir, str(level), str(ty), "{}_{}.dem".format(ty, tx)
                )
                if os.path.exists(tile_dem_path):
                    continue

                pos = Pos(n=level, x=tx, y=ty)
                values = self._sample_tile_dem(pos, p2n, img_gx0, img_gy0, tile_levels)
                self._write_dem_tile(dem_dir, pos, values)


class _StudyTilingProxy:
    """Minimal proxy providing generate_populated_positions for DEM generation."""

    def __init__(self, tile_levels, img_gx0, img_gy0, img_width, img_height, p2n):
        self._tile_levels = tile_levels
        self._img_gx0 = img_gx0
        self._img_gy0 = img_gy0
        self._width = img_width
        self._height = img_height
        self._p2n = p2n

    def generate_populated_positions(self):
        n = self._tile_levels
        img_gx1 = self._img_gx0 + self._width - 1
        img_gy1 = self._img_gy0 + self._height - 1

        tile_start_tx = self._img_gx0 // 256
        tile_start_ty = self._img_gy0 // 256
        tile_end_tx = img_gx1 // 256
        tile_end_ty = img_gy1 // 256

        for ity in range(tile_start_ty, tile_end_ty + 1):
            for itx in range(tile_start_tx, tile_end_tx + 1):
                yield (Pos(n, itx, ity),)
