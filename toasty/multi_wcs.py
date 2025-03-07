# -*- mode: python; coding: utf-8 -*-
# Copyright 2020-2021 the AAS WorldWide Telescope project
# Licensed under the MIT License.

"""
Generate tiles from a collection of images with associated WCS coordinate
systems.

This module has the following Python package dependencies:

- astropy
- reproject
- shapely (to optimize the projection in reproject)
"""

__all__ = '''
MultiWcsProcessor
'''.split()

import numpy as np
from tqdm import tqdm
import warnings

from .image import Image, ImageDescription, ImageMode
from .study import StudyTiling


class MultiWcsDescriptor(object):
    ident = None
    in_shape = None
    in_wcs = None

    imin = None
    imax = None
    jmin = None
    jmax = None

    sub_tiling = None


class MultiWcsProcessor(object):
    def __init__(self, collection):
        self._collection = collection


    def compute_global_pixelization(self, builder):
        from reproject.mosaicking.wcs_helpers import find_optimal_celestial_wcs

        # Load up current WCS information for all of the inputs

        def create_mwcs_descriptor(coll_desc):
            desc = MultiWcsDescriptor()
            desc.ident = coll_desc.collection_id
            desc.in_shape = coll_desc.shape
            desc.in_wcs = coll_desc.wcs
            return desc

        self._descs = [create_mwcs_descriptor(d) for d in self._collection.descriptions()]

        # Compute the optimal tangential tiling that fits all of them. Since WWT
        # tilings must be done in a negative-parity coordinate system, we use an
        # ImageDescription helper to ensure we get that.

        wcs, shape = find_optimal_celestial_wcs(
            ((desc.in_shape, desc.in_wcs) for desc in self._descs),
            auto_rotate = True,
            projection = 'TAN',
        )

        desc = ImageDescription(wcs=wcs, shape=shape)
        desc.ensure_negative_parity()
        self._combined_wcs = desc.wcs
        self._combined_shape = desc.shape
        height, width = self._combined_shape

        self._tiling = StudyTiling(width, height)
        self._tiling.apply_to_imageset(builder.imgset)
        builder.apply_wcs_info(self._combined_wcs, width, height)

        # While we're here, figure out how each input will map onto the global
        # tiling. This makes sure that nothing funky happened during the
        # computation and allows us to know how many tiles we'll have to visit.

        self._n_todo = 0

        for desc in self._descs:
            # XXX: this functionality is largely copied from
            # `reproject.mosaicking.coadd.reproject_and_coadd`, and redundant
            # with it, but it's sufficiently different that I think the best
            # approach is to essentially fork the implementation.

            # Figure out where this array lands in the mosaic.

            ny, nx = desc.in_shape
            xc = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
            yc = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
            xc_out, yc_out = self._combined_wcs.world_to_pixel(desc.in_wcs.pixel_to_world(xc, yc))

            if np.any(np.isnan(xc_out)) or np.any(np.isnan(yc_out)):
                raise Exception(f'segment {desc.ident} does not fit within the global mosaic')

            desc.imin = max(0, int(np.floor(xc_out.min() + 0.5)))
            desc.imax = min(self._combined_shape[1], int(np.ceil(xc_out.max() + 0.5)))
            desc.jmin = max(0, int(np.floor(yc_out.min() + 0.5)))
            desc.jmax = min(self._combined_shape[0], int(np.ceil(yc_out.max() + 0.5)))

            # Compute the sub-tiling now so that we can count how many total
            # tiles we'll need to process.

            if desc.imax < desc.imin or desc.jmax < desc.jmin:
                raise Exception(f'segment {desc.ident} maps to zero size in the global mosaic')

            desc.sub_tiling = self._tiling.compute_for_subimage(
                desc.imin,
                desc.jmin,
                desc.imax - desc.imin,
                desc.jmax - desc.jmin,
            )

            self._n_todo += desc.sub_tiling.count_populated_positions()

        return self  # chaining convenience


    def tile(self, pio, reproject_function, parallel=None, cli_progress=False, **kwargs):
        """
        Tile!!!!

        Parameters
        ----------
        pio : :class:`toasty.pyramid.PyramidIO`
            A :class:`~toasty.pyramid.PyramidIO` instance to manage the I/O with
            the tiles in the tile pyramid.
        reproject_function : TKTK
            TKTK
        parallel : integer or None (the default)
            The level of parallelization to use. If unspecified, defaults to using
            all CPUs. If the OS does not support fork-based multiprocessing,
            parallel processing is not possible and serial processing will be
            forced. Pass ``1`` to force serial processing.
        cli_progress : optional boolean, defaults False
            If true, a progress bar will be printed to the terminal using tqdm.

        """
        from .par_util import resolve_parallelism
        parallel = resolve_parallelism(parallel)

        if parallel > 1:
            self._tile_parallel(pio, reproject_function, cli_progress, parallel, **kwargs)
        else:
            self._tile_serial(pio, reproject_function, cli_progress, **kwargs)

        # Since we used `pio.update_image()`, we should clean up the lockfiles
        # that were generated.
        pio.clean_lockfiles(self._tiling._tile_levels)


    def _tile_serial(self, pio, reproject_function, cli_progress, **kwargs):
        invert_into_tiles = pio.get_default_vertical_parity_sign() == 1

        with tqdm(total=self._n_todo, disable=not cli_progress) as progress:
            for image, desc in zip(self._collection.images(), self._descs):
                # XXX: more copying from
                # `reproject.mosaicking.coadd.reproject_and_coadd`.

                wcs_out_indiv = self._combined_wcs[desc.jmin:desc.jmax, desc.imin:desc.imax]
                shape_out_indiv = (desc.jmax - desc.jmin, desc.imax - desc.imin)

                array = reproject_function(
                    (image.asarray(), image.wcs),
                    output_projection=wcs_out_indiv,
                    shape_out=shape_out_indiv,
                    return_footprint=False,
                    **kwargs
                )

                image = Image.from_array(array.astype(np.float32))

                for pos, width, height, image_x, image_y, tile_x, tile_y in desc.sub_tiling.generate_populated_positions():
                    # Because we are doing an arbitrary WCS reprojection anyway,
                    # we can ensure that our source image is stored with a
                    # top-down vertical data layout, AKA negative image parity,
                    # which is what the overall "study" coordinate system needs.
                    # But if we're writing to FITS format tiles, those need to
                    # end up with a bottoms-up format. So we need to flip the
                    # vertical orientation of how we put the data into the tile
                    # buffer.

                    if invert_into_tiles:
                        flip_tile_y1 = 255 - tile_y
                        flip_tile_y0 = flip_tile_y1 - height

                        if flip_tile_y0 == -1:
                            flip_tile_y0 = None  # with a slice, -1 does the wrong thing

                        by_idx = slice(flip_tile_y1, flip_tile_y0, -1)
                    else:
                        by_idx = slice(tile_y, tile_y + height)

                    iy_idx = slice(image_y, image_y + height)
                    ix_idx = slice(image_x, image_x + width)
                    bx_idx = slice(tile_x, tile_x + width)

                    with pio.update_image(pos, masked_mode=image.mode, default='masked') as basis:
                        image.update_into_maskable_buffer(basis, iy_idx, ix_idx, by_idx, bx_idx)

                    progress.update(1)

        if cli_progress:
            print()


    def _tile_parallel(self, pio, reproject_function, cli_progress, parallel, **kwargs):
        import multiprocessing as mp

        # Start up the workers

        queue = mp.Queue(maxsize = 2 * parallel)
        workers = []

        for _ in range(parallel):
            w = mp.Process(target=_mp_tile_worker, args=(queue, pio, reproject_function, kwargs))
            w.daemon = True
            w.start()
            workers.append(w)

        # Send out them segments

        with tqdm(total=len(self._descs), disable=not cli_progress) as progress:
            for image, desc in zip(self._collection.images(), self._descs):
                wcs_out_indiv = self._combined_wcs[desc.jmin:desc.jmax, desc.imin:desc.imax]
                queue.put((image, desc, wcs_out_indiv))
                progress.update(1)

            queue.close()

            for w in workers:
                w.join()

        if cli_progress:
            print()


def _mp_tile_worker(queue, pio, reproject_function, kwargs):
    """
    Generate and enqueue the tiles that need to be processed.
    """
    from queue import Empty

    invert_into_tiles = pio.get_default_vertical_parity_sign() == 1

    while True:
        try:
            # un-pickling WCS objects always triggers warnings right now
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                image, desc, wcs_out_indiv = queue.get(True, timeout=1)
        except (OSError, ValueError, Empty):
            # OSError or ValueError => queue closed. This signal seems not to
            # cross multiprocess lines, though.
            break

        shape_out_indiv = (desc.jmax - desc.jmin, desc.imax - desc.imin)

        array = reproject_function(
            (image.asarray(), image.wcs),
            output_projection=wcs_out_indiv,
            shape_out=shape_out_indiv,
            return_footprint=False,
            **kwargs
        )

        image = Image.from_array(array.astype(np.float32))

        for pos, width, height, image_x, image_y, tile_x, tile_y in desc.sub_tiling.generate_populated_positions():
            if invert_into_tiles:
                flip_tile_y1 = 255 - tile_y
                flip_tile_y0 = flip_tile_y1 - height

                if flip_tile_y0 == -1:
                    flip_tile_y0 = None  # with a slice, -1 does the wrong thing

                by_idx = slice(flip_tile_y1, flip_tile_y0, -1)
            else:
                by_idx = slice(tile_y, tile_y + height)

            iy_idx = slice(image_y, image_y + height)
            ix_idx = slice(image_x, image_x + width)
            bx_idx = slice(tile_x, tile_x + width)

            with pio.update_image(pos, masked_mode=image.mode, default='masked') as basis:
                image.update_into_maskable_buffer(basis, iy_idx, ix_idx, by_idx, bx_idx)
