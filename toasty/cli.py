# -*- mode: python; coding: utf-8 -*-
# Copyright 2019-2021 the AAS WorldWide Telescope project.
# Licensed under the MIT License.

"""Entrypoint for the "toasty" command-line interface.

"""

from __future__ import absolute_import, division, print_function

__all__ = """
die
entrypoint
warn
""".split()

import argparse
import os.path
import sys
import warnings

from wwt_data_formats.cli import EnsureGlobsExpandedAction


# General CLI utilities


def die(msg):
    print("error:", msg, file=sys.stderr)
    sys.exit(1)


def warn(msg):
    print("warning:", msg, file=sys.stderr)


# "cascade" subcommand


def cascade_getparser(parser):
    parser.add_argument(
        "--parallelism",
        "-j",
        metavar="COUNT",
        type=int,
        help="The parallelization level (default: use all CPUs; specify `1` to force serial processing)",
    )
    parser.add_argument(
        "--format",
        "-f",
        metavar="FORMAT",
        default=None,
        choices=["png", "jpg", "npy", "fits"],
        help="The format of data files to cascade. If not specified, this will be guessed.",
    )
    parser.add_argument(
        "--start",
        metavar="DEPTH",
        type=int,
        help="The depth of the TOAST layer to start the cascade",
    )
    parser.add_argument(
        "pyramid_dir",
        metavar="DIR",
        help="The directory containing the tile pyramid to cascade",
    )


def cascade_impl(settings):
    from .merge import averaging_merger, cascade_images
    from .pyramid import PyramidIO

    pio = PyramidIO(settings.pyramid_dir, default_format=settings.format)

    start = settings.start
    if start is None:
        die("currently, you must specify the start layer with the --start option")

    cascade_images(
        pio, start, averaging_merger, parallel=settings.parallelism, cli_progress=True
    )


# "check-avm" subcommand


def check_avm_getparser(parser):
    parser.add_argument(
        "--print",
        "-p",
        dest="print_avm",
        action="store_true",
        help="Print the AVM data if present",
    )
    parser.add_argument(
        "--exitcode",
        action="store_true",
        help="Exit with code 1 if no AVM tags are present",
    )
    parser.add_argument(
        "image",
        metavar="PATH",
        help="An image to check for AVM tags",
    )


def check_avm_impl(settings):
    from .image import ImageLoader

    try:
        from pyavm import AVM, exceptions

        SYNTAX_EXCEPTIONS = (
            exceptions.AVMListLengthError,
            exceptions.AVMItemNotInControlledVocabularyError,
            exceptions.AVMEmptyValueError,
        )
    except ImportError:
        die("cannot check AVM tags: you must install the `pyavm` package")

    try:
        # PyAVM can issue a lot of warnings that aren't helpful to the user.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            avm = AVM.from_image(settings.image)
    except exceptions.NoXMPPacketFound:
        print(f"{settings.image}: no AVM tags")
        sys.exit(1 if settings.exitcode else 0)
    except SYNTAX_EXCEPTIONS as e:
        print(
            f"{settings.image}: AVM data are present but malformed: {e} ({e.__class__.__name__})"
        )
        sys.exit(1)
    except Exception as e:
        die(
            f"error probing AVM tags of `{settings.image}`: {e} ({e.__class__.__name__})"
        )

    status_exitcode = 0

    if settings.print_avm:
        print_remark = ""
    else:
        print_remark = " (use `--print` to display them)"

    print(f"{settings.image}: AVM tags are present{print_remark}")

    if repr(avm.Spatial) == "":
        print(f"{settings.image}: however, no spatial information")
        status_exitcode = 1
    elif avm.Spatial.ReferenceDimension is None:
        print(
            f"{settings.image}: including spatial information, but no reference dimension"
        )
    else:
        img = ImageLoader().load_path(settings.image)
        this_shape = (int(img.width), int(img.height))
        ref_shape = (
            int(avm.Spatial.ReferenceDimension[0]),
            int(avm.Spatial.ReferenceDimension[1]),
        )

        if this_shape == ref_shape:
            print(
                f"{settings.image}: including spatial information for this image's dimensions"
            )
        else:
            print(
                f"{settings.image}: including spatial information at a different scale"
            )
            print(
                f"{settings.image}: AVM is for {ref_shape[0]} x {ref_shape[1]}, image is {this_shape[0]} x {this_shape[1]}"
            )
            this_aspect = this_shape[0] / this_shape[1]
            ref_aspect = ref_shape[0] / ref_shape[1]

            # The threshold here approximates the one used by pyavm
            if abs(this_aspect - ref_aspect) / (this_aspect + ref_aspect) >= 0.005:
                print(
                    f"{settings.image}: WARNING: these aspect ratios are not compatible"
                )
                status_exitcode = 1

    if settings.print_avm:
        print()
        print(avm)

    sys.exit(status_exitcode if settings.exitcode else 0)


# "make_thumbnail" subcommand


def make_thumbnail_getparser(parser):
    from .image import ImageLoader

    ImageLoader.add_arguments(parser)

    parser.add_argument(
        "imgpath",
        metavar="IN-PATH",
        help="The image file to be thumbnailed",
    )
    parser.add_argument(
        "outpath",
        metavar="OUT-PATH",
        help="The location of the new thumbnail file",
    )


def make_thumbnail_impl(settings):
    from .image import ImageLoader

    olp = settings.outpath.lower()
    if not (olp.endswith(".jpg") or olp.endswith(".jpeg")):
        warn(
            'saving output in JPEG format even though filename is "{}"'.format(
                settings.outpath
            )
        )

    img = ImageLoader.create_from_args(settings).load_path(settings.imgpath)
    thumb = img.make_thumbnail_bitmap()

    with open(settings.outpath, "wb") as f:
        thumb.save(f, format="JPEG")


# "show" subcommand


def show_getparser(parser):
    subparsers = parser.add_subparsers(dest="show_command")
    _parser = subparsers.add_parser("concept-doi")
    _parser = subparsers.add_parser("version")
    _parser = subparsers.add_parser("version-doi")


def show_impl(settings):
    if settings.show_command is None:
        print('Run the "show" command with `--help` for help on its subcommands')
        return

    if settings.show_command == "concept-doi":
        # This string constant will be rewritten by Cranko during releases:
        doi = "xx.xxxx/dev-build.toasty.concept"
        if not doi.startswith("10."):
            warn("this DOI is a fake value used for development builds")
        print(doi)
    elif settings.show_command == "version":
        # This string constant will be rewritten by Cranko during releases:
        version = "0.dev0"  # cranko project-version
        print(version)
    elif settings.show_command == "version-doi":
        # This string constant will be rewritten by Cranko during releases:
        doi = "xx.xxxx/dev-build.toasty.version"
        if not doi.startswith("10."):
            warn("this DOI is a fake value used for development builds")
        print(doi)
    else:
        die('unrecognized "show" subcommand ' + settings.show_command)


# "tile_allsky" subcommand


def tile_allsky_getparser(parser):
    from .image import ImageLoader

    ImageLoader.add_arguments(parser)

    parser.add_argument(
        "--name",
        metavar="NAME",
        default="Toasty",
        help="The image name to embed in the output WTML file (default: %(default)s)",
    )
    parser.add_argument(
        "--outdir",
        metavar="PATH",
        default=".",
        help="The root directory of the output tile pyramid (default: %(default)s)",
    )
    parser.add_argument(
        "--placeholder-thumbnail",
        action="store_true",
        help="Do not attempt to thumbnail the input image -- saves memory for large inputs",
    )
    parser.add_argument(
        "--projection",
        metavar="PROJTYPE",
        default="plate-carree",
        help="The projection type of the input image (default: %(default)s; choices: %(choices)s)",
        choices=[
            "plate-carree",
            "plate-carree-galactic",
            "plate-carree-ecliptic",
            "plate-carree-planet",
            "plate-carree-planet-zeroleft",
            "plate-carree-planet-zeroright",
            "plate-carree-panorama",
        ],
    )
    parser.add_argument(
        "--parallelism",
        "-j",
        metavar="COUNT",
        type=int,
        help="The parallelization level (default: use all CPUs if OS supports; specify `1` to force serial processing)",
    )
    parser.add_argument(
        "imgpath",
        metavar="PATH",
        help="The image file to be tiled",
    )
    parser.add_argument(
        "depth",
        metavar="DEPTH",
        type=int,
        help="The depth of the TOAST layer to sample",
    )


def tile_allsky_impl(settings):
    from .builder import Builder
    from .image import ImageLoader
    from .pyramid import PyramidIO

    img = ImageLoader.create_from_args(settings).load_path(settings.imgpath)
    pio = PyramidIO(settings.outdir)
    is_planet = False
    is_pano = False

    if settings.projection == "plate-carree":
        from .samplers import plate_carree_sampler

        sampler = plate_carree_sampler(img.asarray())
    elif settings.projection == "plate-carree-galactic":
        from .samplers import plate_carree_galactic_sampler

        sampler = plate_carree_galactic_sampler(img.asarray())
    elif settings.projection == "plate-carree-ecliptic":
        from .samplers import plate_carree_ecliptic_sampler

        sampler = plate_carree_ecliptic_sampler(img.asarray())
    elif settings.projection == "plate-carree-planet":
        from .samplers import plate_carree_planet_sampler

        sampler = plate_carree_planet_sampler(img.asarray())
        is_planet = True
    elif settings.projection == "plate-carree-planet-zeroleft":
        from .samplers import plate_carree_planet_zeroleft_sampler

        sampler = plate_carree_planet_zeroleft_sampler(img.asarray())
        is_planet = True
    elif settings.projection == "plate-carree-planet-zeroright":
        from .samplers import plate_carree_zeroright_sampler

        sampler = plate_carree_zeroright_sampler(img.asarray())
        is_planet = True
    elif settings.projection == "plate-carree-panorama":
        from .samplers import plate_carree_sampler

        sampler = plate_carree_sampler(img.asarray())
        is_pano = True
    else:
        die(
            "the image projection type {!r} is not recognized".format(
                settings.projection
            )
        )

    builder = Builder(pio)

    # Do the thumbnail first since for large inputs it can be the memory high-water mark!
    if settings.placeholder_thumbnail:
        builder.make_placeholder_thumbnail()
    else:
        builder.make_thumbnail_from_other(img)

    builder.toast_base(
        sampler,
        settings.depth,
        is_planet=is_planet,
        is_pano=is_pano,
        parallel=settings.parallelism,
        cli_progress=True,
    )
    builder.set_name(settings.name)
    builder.write_index_rel_wtml()

    print(
        f'Successfully tiled input "{settings.imgpath}" at level {builder.imgset.tile_levels}.'
    )
    print("To create parent tiles, consider running:")
    print()
    print(f"   toasty cascade --start {builder.imgset.tile_levels} {settings.outdir}")


# "tile_healpix" subcommand


def tile_healpix_getparser(parser):
    parser.add_argument(
        "--galactic",
        action="store_true",
        help="Force use of Galactic coordinate system, regardless of headers",
    )
    parser.add_argument(
        "--outdir",
        metavar="PATH",
        default=".",
        help="The root directory of the output tile pyramid",
    )
    parser.add_argument(
        "--parallelism",
        "-j",
        metavar="COUNT",
        type=int,
        help="The parallelization level (default: use all CPUs if OS supports; specify `1` to force serial processing)",
    )
    parser.add_argument(
        "fitspath",
        metavar="PATH",
        help="The HEALPix FITS file to be tiled",
    )
    parser.add_argument(
        "depth",
        metavar="DEPTH",
        type=int,
        help="The depth of the TOAST layer to sample",
    )


def tile_healpix_impl(settings):
    from .builder import Builder
    from .pyramid import PyramidIO
    from .samplers import healpix_fits_file_sampler

    pio = PyramidIO(settings.outdir, default_format="fits")
    sampler = healpix_fits_file_sampler(
        settings.fitspath, force_galactic=settings.galactic
    )
    builder = Builder(pio)
    builder.toast_base(
        sampler,
        settings.depth,
        parallel=settings.parallelism,
        cli_progress=True,
    )
    builder.write_index_rel_wtml()

    print(
        f'Successfully tiled input "{settings.fitspath}" at level {builder.imgset.tile_levels}.'
    )
    print("To create parent tiles, consider running:")
    print()
    print(f"   toasty cascade --start {builder.imgset.tile_levels} {settings.outdir}")


# "tile_multi_tan" subcommand


def tile_multi_tan_getparser(parser):
    parser.add_argument(
        "--parallelism",
        "-j",
        metavar="COUNT",
        type=int,
        help="The parallelization level (default: use all CPUs; specify `1` to force serial processing)",
    )
    parser.add_argument(
        "--hdu-index",
        metavar="INDEX",
        type=int,
        default=0,
        help="Which HDU to load in each input FITS file",
    )
    parser.add_argument(
        "--wcs-key",
        metavar="LETTER",
        default=" ",
        help="Which group of WCS headers to process in the input HDU(s)",
    )
    parser.add_argument(
        "--outdir",
        metavar="PATH",
        default=".",
        help="The root directory of the output tile pyramid",
    )
    parser.add_argument(
        "paths",
        metavar="PATHS",
        action=EnsureGlobsExpandedAction,
        nargs="+",
        help="The FITS files with image data",
    )


def tile_multi_tan_impl(settings):
    from .builder import Builder
    from .collection import SimpleFitsCollection
    from .multi_tan import MultiTanProcessor
    from .pyramid import PyramidIO

    pio = PyramidIO(settings.outdir, default_format="fits")
    builder = Builder(pio)

    collection = SimpleFitsCollection(settings.paths, hdu_index=settings.hdu_index, wcs_key=settings.wcs_key)

    mtp = MultiTanProcessor(collection)
    mtp.compute_global_pixelization(builder)
    mtp.tile(
        pio,
        parallel=settings.parallelism,
        cli_progress=True,
    )
    builder.write_index_rel_wtml()

    print(f"Successfully tiled inputs at level {builder.imgset.tile_levels}.")
    print("To create parent tiles, consider running:")
    print()
    print(f"   toasty cascade --start {builder.imgset.tile_levels} {settings.outdir}")


# "tile_study" subcommand


def tile_study_getparser(parser):
    from .image import ImageLoader

    ImageLoader.add_arguments(parser)

    parser.add_argument(
        "--name",
        metavar="NAME",
        default="Toasty",
        help="The image name to embed in the output WTML file (default: %(default)s)",
    )
    parser.add_argument(
        "--avm",
        action="store_true",
        help="Expect the input image to have AVM positioning tags",
    )
    parser.add_argument(
        "--avm-from",
        metavar="PATH",
        help="Set positioning based on AVM tags in a different image",
    )
    parser.add_argument(
        "--fits-wcs",
        metavar="PATH",
        help="Get WCS information from this FITS file",
    )
    parser.add_argument(
        "--placeholder-thumbnail",
        action="store_true",
        help="Do not attempt to thumbnail the input image -- saves memory for large inputs",
    )
    parser.add_argument(
        "--outdir",
        metavar="PATH",
        default=".",
        help="The root directory of the output tile pyramid",
    )
    parser.add_argument(
        "imgpath",
        metavar="PATH",
        help="The study image file to be tiled",
    )


def tile_study_impl(settings):
    from .builder import Builder
    from .image import ImageLoader
    from .pyramid import PyramidIO

    img = ImageLoader.create_from_args(settings).load_path(settings.imgpath)
    pio = PyramidIO(settings.outdir, default_format=img.default_format)
    builder = Builder(pio)

    # First, prepare the tiling, because we can't correctly apply WCS until we
    # know the details of the tiling parameters

    tiling = builder.prepare_study_tiling(img)

    # Now deal with the WCS

    if settings.avm or settings.avm_from:
        # We don't *always* check for AVM because pyavm prints out a lot of junk
        # and may not be installed.
        try:
            from pyavm import AVM
        except ImportError:
            die("cannot use AVM data: you must install the `pyavm` package")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if settings.avm_from:
                    if settings.avm:
                        print(
                            "warning: `--avm` option superseded by `--avm-from`",
                            file=sys.stderr,
                        )
                    avm_input = settings.avm_from
                else:
                    avm_input = settings.imgpath

                avm = AVM.from_image(avm_input)
        except Exception:
            print(
                f"error: failed to read AVM tags of input `{avm_input}`",
                file=sys.stderr,
            )
            raise

        builder.apply_avm_info(avm, img.width, img.height)
    elif settings.fits_wcs:
        try:
            from astropy.io import fits
            from astropy.wcs import WCS
        except ImportError:
            die("cannot use FITS WCS: you must install the `astropy` package")

        try:
            with fits.open(settings.fits_wcs) as hdul:
                # TODO: more flexibility in HDU choice:
                hdu = hdul[0]
                wcs = WCS(hdu, fobj=hdul)
        except Exception as e:
            die(f"cannot read WCS from FITS file `{settings.fits_wcs}`: {e}")

        wcs_width = hdu.header.get("IMAGEW")  # Astrometry.net
        wcs_height = hdu.header.get("IMAGEH")

        if wcs_width is None and len(hdu.shape) > 1:
            wcs_width = hdu.shape[-1]
            wcs_height = hdu.shape[-2]

        if wcs_width is None:
            warn(
                f"cannot infer image dimensions from WCS FITS file `{settings.fits_wcs}`; "
                f"unable to check consistency with input image"
            )
        elif img.shape[:2] != (wcs_height, wcs_width):
            warn(
                f"image `{settings.imgpath}` has shape {img.shape}, but "
                f"WCS reference file `{settings.fits_wcs}` has shape ({wcs_height}, {wcs_width})"
            )

            # Cribbing some code from pyavm ...
            scale_x = img.shape[1] / wcs_width
            scale_y = img.shape[0] / wcs_height

            if abs(scale_x - scale_y) / (scale_x + scale_y) >= 0.005:
                warn(
                    "the aspect ratios are not compatible; I'll proceed, but astrometry is unlikely transfer correctly"
                )
                scale = 1.0
            else:
                warn("the aspect ratios are compatible, so I'll try rescaling")
                scale = scale_x

            wcs.wcs.crpix *= scale

            if hasattr(wcs.wcs, "cd"):
                wcs.wcs.cd /= scale
            else:
                wcs.wcs.cdelt /= scale

            if hasattr(wcs, "naxis1"):
                wcs.naxis1 = img.shape[1]
                wcs.naxis2 = img.shape[0]

        img._wcs = wcs  # <= hack alert, but I think this is OK
        img.ensure_negative_parity()
        builder.apply_wcs_info(img.wcs, img.width, img.height)
    elif img.wcs is not None:
        # Study images must have negative parity.
        img.ensure_negative_parity()
        builder.apply_wcs_info(img.wcs, img.width, img.height)
    else:
        builder.default_tiled_study_astrometry()

    # Do the thumbnail before the main image since for large inputs it can be
    # the memory high-water mark!
    if settings.placeholder_thumbnail:
        builder.make_placeholder_thumbnail()
    else:
        builder.make_thumbnail_from_other(img)

    # Finally, actually tile the image
    builder.execute_study_tiling(img, tiling, cli_progress=True)

    # Final metadata
    builder.set_name(settings.name)
    builder.write_index_rel_wtml()

    print(
        f'Successfully tiled input "{settings.imgpath}" at level {builder.imgset.tile_levels}.'
    )
    print("To create parent tiles, consider running:")
    print()
    print(f"   toasty cascade --start {builder.imgset.tile_levels} {settings.outdir}")


# "tile_planet_study" subcommand


def tile_planet_study_getparser(parser):
    parser.add_argument(
        "--name",
        metavar="NAME",
        default="Toasty",
        help="The image name to embed in the output WTML (default: %(default)s)",
    )
    parser.add_argument(
        "--placeholder-thumbnail",
        action="store_true",
        help="Skip thumbnailing the input (saves memory for very large inputs)",
    )
    parser.add_argument(
        "--radius",
        metavar="METERS",
        type=float,
        help="Body radius (default: the semi-major axis of the input's CRS)",
    )
    parser.add_argument(
        "--resolution",
        metavar="METERS",
        type=float,
        help="Output pixel size (default: the input's native resolution)",
    )
    parser.add_argument(
        "--outdir",
        metavar="PATH",
        default=".",
        help="The root directory of the output tile pyramid (default: %(default)s)",
    )
    parser.add_argument(
        "image_path",
        metavar="IMAGE",
        help="Georeferenced planetary raster (GeoTIFF, PDS .IMG, ISIS cube, or anything GDAL reads)",
    )


def tile_planet_study_impl(settings):
    from .geo_tile import PlanetStudyTiler

    tiler = PlanetStudyTiler(settings.image_path, radius_m=settings.radius, resolution_m=settings.resolution)
    builder = tiler.tile(
        settings.outdir,
        name=settings.name,
        placeholder_thumbnail=settings.placeholder_thumbnail,
        cli_progress=True,
    )
    imgset = builder.imgset
    print(f'Successfully tiled "{settings.image_path}" at level {imgset.tile_levels}.')
    print(
        f"Center: {tiler.center_lat:.6f}° lat, {tiler.center_lon:.6f}° E lon; "
        f"{tiler.resolution_m:.3f} m/px; {tiler.width}x{tiler.height} px"
    )
    print("To generate parent tiles, run:")
    print()
    print(f"   toasty cascade --start {imgset.tile_levels} {settings.outdir}")
    print()
    print("To add elevation, run tile-planet-dem with a DEM and this pyramid's index_rel.wtml.")


# "tile_planet_dem" subcommand


def tile_planet_dem_getparser(parser):
    parser.add_argument(
        "--offset",
        metavar="METERS",
        type=float,
        default=0.0,
        help="Subtract this from every DEM sample, e.g. the body radius for DEMs that store radius (default: 0)",
    )
    parser.add_argument(
        "dem_path",
        metavar="DEM",
        help="Elevation raster in meters (GeoTIFF, PDS .IMG, or anything GDAL reads)",
    )
    parser.add_argument(
        "wtml_path",
        metavar="WTML",
        help="The index_rel.wtml written by tile-planet-study",
    )


def tile_planet_dem_impl(settings):
    from .geo_tile import PlanetDemTiler

    tiler = PlanetDemTiler(settings.dem_path, settings.wtml_path, offset_m=settings.offset)
    n = tiler.tile(cli_progress=True)
    print(f"Wrote {n} DEM tiles and updated {settings.wtml_path}.")


# "tile_wwtl" subcommand


def tile_wwtl_getparser(parser):
    from .image import ImageLoader

    ImageLoader.add_arguments(parser)

    parser.add_argument(
        "--name",
        metavar="NAME",
        default="Toasty",
        help="The image name to embed in the output WTML file (default: %(default)s)",
    )
    parser.add_argument(
        "--placeholder-thumbnail",
        action="store_true",
        help="Do not attempt to thumbnail the input image -- saves memory for large inputs",
    )
    parser.add_argument(
        "--outdir",
        metavar="PATH",
        default=".",
        help="The root directory of the output tile pyramid",
    )
    parser.add_argument(
        "wwtl_path",
        metavar="WWTL-PATH",
        help="The WWTL layer file to be processed",
    )


def tile_wwtl_impl(settings):
    from .builder import Builder
    from .pyramid import PyramidIO

    pio = PyramidIO(settings.outdir)
    builder = Builder(pio)
    img = builder.load_from_wwtl(settings, settings.wwtl_path, cli_progress=True)

    # Do the thumbnail first since for large inputs it can be the memory high-water mark!
    if settings.placeholder_thumbnail:
        builder.make_placeholder_thumbnail()
    else:
        builder.make_thumbnail_from_other(img)

    builder.set_name(settings.name)
    builder.write_index_rel_wtml()

    print(
        f'Successfully tiled input "{settings.wwtl_path}" at level {builder.imgset.tile_levels}.'
    )
    print("To create parent tiles, consider running:")
    print()
    print(f"   toasty cascade --start {builder.imgset.tile_levels} {settings.outdir}")


# "transform" subcommand


def transform_getparser(parser):
    subparsers = parser.add_subparsers(dest="transform_command")

    parser = subparsers.add_parser("fx3-to-rgb")
    parser.add_argument(
        "--parallelism",
        "-j",
        metavar="COUNT",
        type=int,
        help="The parallelization level (default: use all CPUs; specify `1` to force serial processing)",
    )
    parser.add_argument(
        "--start",
        metavar="DEPTH",
        type=int,
        help="How deep in the tile pyramid to proceed wth the transformation",
    )
    parser.add_argument(
        "--clip",
        "-c",
        metavar="NUMBER",
        type=float,
        default=1.0,
        help="The level at which to start flipping the floating-point data",
    )
    parser.add_argument(
        "--outdir",
        metavar="PATH",
        help="Output transformed data into this directory, instead of operating in-place",
    )
    parser.add_argument(
        "pyramid_dir",
        metavar="DIR",
        help="The directory containing the tile pyramid to transform",
    )

    parser = subparsers.add_parser("u8-to-rgb")
    parser.add_argument(
        "--parallelism",
        "-j",
        metavar="COUNT",
        type=int,
        help="The parallelization level (default: use all CPUs; specify `1` to force serial processing)",
    )
    parser.add_argument(
        "--start",
        metavar="DEPTH",
        type=int,
        help="How deep in the tile pyramid to proceed wth the transformation",
    )
    parser.add_argument(
        "--outdir",
        metavar="PATH",
        help="Output transformed data into this directory, instead of operating in-place",
    )
    parser.add_argument(
        "pyramid_dir",
        metavar="DIR",
        help="The directory containing the tile pyramid to transform",
    )


def transform_impl(settings):
    from .pyramid import PyramidIO

    if settings.transform_command is None:
        print('Run the "transform" command with `--help` for help on its subcommands')
        return

    if settings.transform_command == "fx3-to-rgb":
        pio = PyramidIO(settings.pyramid_dir)
        pio_out = None

        if settings.outdir is not None:
            pio_out = PyramidIO(settings.outdir)

        from .transform import f16x3_to_rgb

        f16x3_to_rgb(
            pio,
            settings.start,
            clip=settings.clip,
            pio_out=pio_out,
            parallel=settings.parallelism,
            cli_progress=True,
        )
    elif settings.transform_command == "u8-to-rgb":
        pio = PyramidIO(settings.pyramid_dir)
        pio_out = None

        if settings.outdir is not None:
            pio_out = PyramidIO(settings.outdir)

        from .transform import u8_to_rgb

        u8_to_rgb(
            pio,
            settings.start,
            pio_out=pio_out,
            parallel=settings.parallelism,
            cli_progress=True,
        )
    else:
        die('unrecognized "transform" subcommand ' + settings.transform_command)


# "view" subcommand
#
# This largely duplicates `wwtdatatool preview` in the `wwt_data_formats`
# package. The difference is that we automagically tile and output the
# `index_rel.wtml`, while `wwtdatatool` requires it to be preexisting.


def view_getparser(parser):
    from .collection import CollectionLoader

    CollectionLoader.add_arguments(parser)

    parser.add_argument(
        "--parallelism",
        "-j",
        metavar="COUNT",
        type=int,
        help="The parallelization level (default: use all CPUs; specify `1` to force serial processing)",
    )
    parser.add_argument(
        "--tunnel",
        "-t",
        metavar="HOST",
        help="Use SSH tunneling to view an image stored on a remote host",
    )
    parser.add_argument(
        "--tunnel-initcmd",
        metavar="COMMAND",
        help="Execute this line of shell script before tunneling processing",
    )
    parser.add_argument(
        "--browser",
        "-b",
        metavar="BROWSER-TYPE",
        help="The type of browser to use for the viewer (as per Python webbrowser)",
    )
    parser.add_argument(
        "--appurl",
        metavar="URL",
        help="The URL of the app to use (intended for developers)",
    )
    parser.add_argument(
        "--tile-only",
        action="store_true",
        help="Tile the data but do not open for viewing",
    )
    parser.add_argument(
        "--tiling-method",
        default="auto",
        choices=["auto", "tan", "toast", "hips"],
        help="The target projection when tiling: `auto`, `tan`, `toast`, or `hips`",
    )
    parser.add_argument(
        "paths",
        metavar="PATHS",
        action=EnsureGlobsExpandedAction,
        nargs="+",
        help="The FITS file(s) with image data",
    )


def view_locally(settings):
    from wwt_data_formats.server import preview_wtml
    from . import TilingMethod
    from .collection import CollectionLoader
    from .fits_tiler import FitsTiler

    if settings.tiling_method == "auto":
        tiling_method = TilingMethod.AUTO_DETECT
    elif settings.tiling_method == "tan":
        tiling_method = TilingMethod.TAN
    elif settings.tiling_method == "toast":
        tiling_method = TilingMethod.TOAST
    elif settings.tiling_method == "hips":
        tiling_method = TilingMethod.HIPS
    else:
        # This shouldn't happen since argparse should validate the input
        die(f"unhandled tiling method `{settings.tiling_method}`")

    coll = CollectionLoader.create_from_args(settings).load_paths(settings.paths)

    # Ignore any astropy WCS/FITS warnings, which can spew a lot of annoying output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tiler = FitsTiler(coll, tiling_method=tiling_method)
        tiler.tile(cli_progress=True, parallel=settings.parallelism)

    rel_wtml_path = os.path.join(tiler.out_dir, "index_rel.wtml")

    if settings.tile_only:
        print("WTML:", rel_wtml_path)
        return

    preview_wtml(
        rel_wtml_path,
        browser=settings.browser,
        app_type="research",
        app_url=settings.appurl,
    )


def view_tunneled(settings):
    import random
    import shlex
    import subprocess
    from urllib.parse import urlparse, urlunparse
    from wwt_data_formats.folder import Folder
    from wwt_data_formats.server import launch_app_for_wtml

    # First: tunnel to the remote host to tile the data and get the path to the
    # index_rel.wtml.
    #
    # If you do `ssh $host $command`, you don't get a login shell, and in some
    # setups that means that not all environment initialization happens. Which
    # might mean that the `toasty` command won't be available. We work around
    # that by avoiding $command and writing an `exec` to stdin. This is gross
    # and raises issues of escaping funky filenames, but those issues already
    # exist with SSH command arguments.
    #
    # We give SSH `-T` to prevent warnings about not allocating pseudo-TTYs.

    ssh_argv = ["ssh", "-T", settings.tunnel]
    toasty_argv = [
        "exec",
        "toasty",
        "view",
        "--tile-only",
        f"--tiling-method={settings.tiling_method}",
    ]

    if settings.parallelism:
        toasty_argv += ["--parallelism", str(settings.parallelism)]
    toasty_argv += [shlex.quote(p) for p in settings.paths]

    print(f"Preparing data on `{settings.tunnel}` ...\n")

    proc = subprocess.Popen(
        ssh_argv,
        shell=False,
        close_fds=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if settings.tunnel_initcmd:
        print(settings.tunnel_initcmd, file=proc.stdin)

    print(" ".join(toasty_argv), file=proc.stdin)
    proc.stdin.close()

    index_rel_path = None

    for line in proc.stdout:
        print(line, end="")

        if line.startswith("WTML:"):
            index_rel_path = line.rstrip().split(None, 1)[1]

    proc.wait(timeout=180)
    if proc.returncode:
        die(f"SSH command exited with error code {proc.returncode}")

    if not index_rel_path:
        die("SSH command seemed successful, but did not return path of data WTML file")

    del proc

    # Next, launch the server. Same environment workaround as before. We use a
    # special "heartbeat" mode in the server to get it to exit reliably when the
    # SSH connection goes away. `--port 0` has the OS automatically choose a
    # free port.

    serve_argv = [
        "exec",
        "wwtdatatool",
        "serve",
        "--port",
        "0",
        "--heartbeat",
        shlex.quote(os.path.dirname(index_rel_path)),
    ]

    if settings.tunnel_initcmd:
        serve_argv = [settings.tunnel_initcmd, ";"] + serve_argv

    print(f"\nLaunching data server on `{settings.tunnel}` ...\n")
    serve_proc = None

    try:
        serve_proc = subprocess.Popen(
            ssh_argv,
            shell=False,
            close_fds=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        print(" ".join(serve_argv), file=serve_proc.stdin, flush=True)
        serve_proc.stdin.close()
        remote_serve_url = None

        for line in serve_proc.stdout:
            print(line, end="")

            if line.startswith("listening at:"):
                remote_serve_url = line.rstrip().split(None, 2)[2]
                break

        if serve_proc.poll() is not None:
            die(f"SSH command exited early with code {serve_proc.returncode}")

        if not remote_serve_url:
            die("SSH command did not return web service URL")

        try:
            remote_url_parsed = urlparse(remote_serve_url)
        except Exception as e:
            die(f"could not parse service URL `{remote_serve_url}`: {e}")

        # Now forward the relevant port. Unfortunately it does not seem to be
        # possible to have SSH auto-choose the local port. It is possible to use a
        # Unix socket on the local side, but then we won't work on non-Unix.

        print("\nLaunching viewer here ...\n")
        port = None

        try:
            for _ in range(10):
                port = random.randint(44000, 48000)
                forward_argv = [
                    "ssh",
                    "-O",
                    "forward",
                    "-L",
                    f"{port}:{remote_url_parsed.netloc}",
                    settings.tunnel,
                ]

                try:
                    subprocess.check_call(forward_argv, shell=False)
                    break
                except Exception as e:
                    warn(f"failed to forward port {port}: {e}")
                    port = None

            if port is None:
                die("could not find a local port to relay traffic (?!)")

            print(f"  Local port: {port}")

            # Somewhat annoyingly, for the research app we currently need to get
            # the URL of the imageset. The most reliable way to do that is going
            # to be to look at the WTML. This reproduces some of the logic in
            # `wwt_data_formats.server.preview_wtml()`.

            wtml_base = os.path.basename(index_rel_path).replace("_rel.wtml", ".wtml")
            local_wtml_url = urlunparse(
                remote_url_parsed._replace(
                    netloc=f"127.0.0.1:{port}",
                    path=f"{remote_url_parsed.path}{wtml_base}",
                )
            )
            fld = Folder.from_url(local_wtml_url)
            image_url = None

            for _index, _kind, imgset in fld.immediate_imagesets():
                image_url = imgset.url
                break

            if image_url is None:
                die("found no imagesets in data WTML folder `{local_wtml_url}`")

            # Now we can finally launch the local browser.

            print("  Local WTML:", local_wtml_url)

            desc = launch_app_for_wtml(
                local_wtml_url,
                image_url=image_url,
                browser=settings.browser,
                app_type="research",
                app_url=settings.appurl,
            )

            print(
                f"\nOpening data in {desc} ... type Control-C to terminate this program when done"
            )

            try:
                # We need to keep on reading the heartbeats sent by the data server
                # to prevent its buffers from filling up.
                while True:
                    serve_proc.stdout.readline()
            except KeyboardInterrupt:
                print("\n(interrupted)\n")
            except Exception as e:
                print(f"\nTerminated on exception: {e} ({e.__class__.__name__})\n")
        finally:
            # Clean up the port forward.

            if port is not None:
                forward_argv[2] = "cancel"
                try:
                    subprocess.check_call(forward_argv, shell=False)
                except Exception as e:
                    warn(f"failed to cancel port forward: {e}")
    finally:
        # Clean up the server process. With our "heartbeat" mode,
        # this is pretty straightforward.

        if serve_proc is not None:
            serve_proc.stdout.close()
            serve_proc.stdin.close()
            serve_proc.wait()


def view_impl(settings):
    if settings.tunnel:
        view_tunneled(settings)
    else:
        view_locally(settings)


# The CLI driver:


def entrypoint(args=None):
    """The entrypoint for the \"toasty\" command-line interface.

    Parameters
    ----------
    args : iterable of str, or None (the default)
      The arguments on the command line. The first argument should be
      a subcommand name or global option; there is no ``argv[0]``
      parameter.

    """
    # Set up the subcommands. We use locals() and globals() in a fairly gross
    # way to avoid circular import issues in Sphinx.

    from .pipeline.cli import pipeline_getparser, pipeline_impl

    names = dict(locals())
    names.update(globals())

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    commands = set()

    for py_name, value in names.items():
        if py_name.endswith("_getparser"):
            cmd_name = py_name[:-10].replace("_", "-")
            subparser = subparsers.add_parser(cmd_name)
            value(subparser)
            commands.add(cmd_name)

    # What did we get?

    settings = parser.parse_args(args)

    if settings.subcommand is None:
        print("Run me with --help for help. Allowed subcommands are:")
        print()
        for cmd in sorted(commands):
            print("   ", cmd)
        return

    py_name = settings.subcommand.replace("-", "_")

    impl = names.get(py_name + "_impl")
    if impl is None:
        die('no such subcommand "{}"'.format(settings.subcommand))

    # OK to go!

    impl(settings)
