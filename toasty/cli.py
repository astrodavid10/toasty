# -*- mode: python; coding: utf-8 -*-
# Copyright 2019 the AAS WorldWide Telescope project.
# Licensed under the MIT License.

"""Entrypoint for the "toasty" command-line interface.

"""
from __future__ import absolute_import, division, print_function

import argparse
import sys


# General CLI utilities

def die(msg):
    print('error:', msg, file=sys.stderr)
    sys.exit(1)

def warn(msg):
    print('warning:', msg, file=sys.stderr)


def indent_xml(elem, level=0):
    """A dumb XML indenter.

    We create XML files using xml.etree.ElementTree, which is careful about
    spacing and so by default creates ugly files with no linewraps or
    indentation. This function is copied from `ElementLib
    <http://effbot.org/zone/element-lib.htm#prettyprint>`_ and implements
    basic, sensible indentation using "tail" text.

    """
    i = "\n" + level * "  "

    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:  # intentionally updating "elem" here!
            indent_xml(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# "cascade" subcommand

def cascade_getparser(parser):
    parser.add_argument(
        '--start',
        metavar = 'DEPTH',
        type = int,
        help = 'The depth of the TOAST layer to start the cascade',
    )
    parser.add_argument(
        'pyramid_dir',
        metavar = 'DIR',
        help = 'The directory containg the tile pyramid to cascade',
    )


def cascade_impl(settings):
    from .merge import averaging_merger, cascade_images
    from .pyramid import PyramidIO

    pio = PyramidIO(settings.pyramid_dir)

    start = settings.start
    if start is None:
        die('currently, you must specify the start layer with the --start option')

    cascade_images(pio, start, averaging_merger)


# "healpix_sample_data_tiles" subcommand

def healpix_sample_data_tiles_getparser(parser):
    parser.add_argument(
        '--outdir',
        metavar = 'PATH',
        default = '.',
        help = 'The root directory of the output tile pyramid',
    )
    parser.add_argument(
        'fitspath',
        metavar = 'PATH',
        help = 'The HEALPix FITS file to be tiled',
    )
    parser.add_argument(
        'depth',
        metavar = 'DEPTH',
        type = int,
        help = 'The depth of the TOAST layer to sample',
    )


def healpix_sample_data_tiles_impl(settings):
    from .pyramid import PyramidIO
    from .samplers import healpix_fits_file_sampler
    from .toast import SamplingToastDataSource

    pio = PyramidIO(settings.outdir)
    sampler = healpix_fits_file_sampler(settings.fitspath)
    ds = SamplingToastDataSource(sampler)
    ds.sample_data_layer(pio, settings.depth)


# "image_sample_tiles" subcommand

def image_sample_tiles_getparser(parser):
    parser.add_argument(
        '--outdir',
        metavar = 'PATH',
        default = '.',
        help = 'The root directory of the output tile pyramid',
    )
    parser.add_argument(
        '--projection',
        metavar = 'PROJTYPE',
        default = 'plate-carree',
        help = 'The projection of the image; "plate-carree" is the only allowed choice',
    )
    parser.add_argument(
        'imgpath',
        metavar = 'PATH',
        help = 'The image file to be tiled',
    )
    parser.add_argument(
        'depth',
        metavar = 'DEPTH',
        type = int,
        help = 'The depth of the TOAST layer to sample',
    )


def image_sample_tiles_impl(settings):
    from .io import read_image
    from .pyramid import PyramidIO
    from .toast import SamplingToastDataSource

    pio = PyramidIO(settings.outdir)
    data = read_image(settings.imgpath)

    if settings.projection == 'plate-carree':
        from .samplers import plate_carree_sampler
        sampler = plate_carree_sampler(data)
    else:
        die('the image projection type {!r} is not recognized'.format(settings.projection))

    ds = SamplingToastDataSource(sampler)
    ds.sample_image_layer(pio, settings.depth)


# "multi_tan_make_data_tiles" subcommand

def multi_tan_make_data_tiles_getparser(parser):
    parser.add_argument(
        '--hdu-index',
        metavar = 'INDEX',
        type = int,
        default = 0,
        help = 'Which HDU to load in each input FITS file',
    )
    parser.add_argument(
        '--outdir',
        metavar = 'PATH',
        default = '.',
        help = 'The root directory of the output tile pyramid',
    )
    parser.add_argument(
        'paths',
        metavar = 'PATHS',
        nargs = '+',
        help = 'The FITS files with image data',
    )

def multi_tan_make_data_tiles_impl(settings):
    from .multi_tan import MultiTanDataSource
    from .pyramid import PyramidIO

    pio = PyramidIO(settings.outdir)
    ds = MultiTanDataSource(settings.paths, hdu_index=settings.hdu_index)
    ds.compute_global_pixelization()

    print('Generating Numpy-formatted data tiles in directory {!r} ...'.format(settings.outdir))
    percentiles = ds.generate_deepest_layer_numpy(pio)

    if len(percentiles):
        print()
        print('Median percentiles in the data:')
        for p in sorted(percentiles.keys()):
            print('   {} = {}'.format(p, percentiles[p]))


# "multi_tan_make_wtml" subcommand

def multi_tan_make_wtml_getparser(parser):
    parser.add_argument(
        '--hdu-index',
        metavar = 'INDEX',
        type = int,
        default = 0,
        help = 'Which HDU to load in each input FITS file',
    )
    parser.add_argument(
        '--name',
        metavar = 'NAME',
        default = 'MultiTan',
        help = 'The dataset name to embed in the WTML file',
    )
    parser.add_argument(
        '--url-prefix',
        metavar = 'PREFIX',
        default = './',
        help = 'The prefix to the tile URL that will be embedded in the WTML',
    )
    parser.add_argument(
        '--fov-factor',
        metavar = 'NUMBER',
        type = float,
        default = 1.7,
        help = 'How tall the FOV should be (ie the zoom level) when viewing this image, in units of the image height',
    )
    parser.add_argument(
        '--bandpass',
        metavar = 'BANDPASS-NAME',
        default = 'Visible',
        help = 'The bandpass of the image data: "Gamma", "HydrogenAlpha", "IR", "Microwave", "Radio", "Ultraviolet", "Visible", "VisibleNight", "XRay"',
    )
    parser.add_argument(
        '--description',
        metavar = 'TEXT',
        default = '',
        help = 'Free text describing what this image is',
    )
    parser.add_argument(
        '--credits-text',
        metavar = 'TEXT',
        default = 'Created by toasty, part of the AAS WorldWide Telescope.',
        help = 'A brief credit of who created and processed the image data',
    )
    parser.add_argument(
        '--credits-url',
        metavar = 'URL',
        default = '',
        help = 'A URL with additional credit information',
    )
    parser.add_argument(
        '--thumbnail-url',
        metavar = 'URL',
        default = '',
        help = 'A URL of a thumbnail image (96x45 JPEG) representing this dataset',
    )
    parser.add_argument(
        'paths',
        metavar = 'PATHS',
        nargs = '+',
        help = 'The FITS files with image data',
    )

def multi_tan_make_wtml_impl(settings):
    from xml.etree import ElementTree as etree
    from .multi_tan import MultiTanDataSource

    ds = MultiTanDataSource(settings.paths, hdu_index=settings.hdu_index)
    ds.compute_global_pixelization()

    folder = ds.create_wtml(
        name = settings.name,
        url_prefix = settings.url_prefix,
        fov_factor = settings.fov_factor,
        bandpass = settings.bandpass,
        description_text = settings.description,
        credits_text = settings.credits_text,
        credits_url = settings.credits_url,
        thumbnail_url = settings.thumbnail_url,
    )
    indent_xml(folder)
    doc = etree.ElementTree(folder)
    doc.write(sys.stdout, encoding='utf-8', xml_declaration=True)


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
    # Set up the subcommands from globals()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    commands = set()

    for py_name, value in globals().items():
        if py_name.endswith('_getparser'):
            cmd_name = py_name[:-10].replace('_', '-')
            subparser = subparsers.add_parser(cmd_name)
            value(subparser)
            commands.add(cmd_name)

    # What did we get?

    settings = parser.parse_args(args)

    if settings.subcommand is None:
        print('Run me with --help for help. Allowed subcommands are:')
        print()
        for cmd in sorted(commands):
            print('   ', cmd)
        return

    py_name = settings.subcommand.replace('-', '_')

    impl = globals().get(py_name + '_impl')
    if impl is None:
        die('no such subcommand "{}"'.format(settings.subcommand))

    # OK to go!

    impl(settings)
