.. _cli/disassemble:

Disassemble one large image to small images
============================================

Assuming you want to split large image into small images::


  $ lumi disassemble --input_dir mosaic/ --fmt tif --tile_size 40, 50 --output_dir split/

The ``lumi disassemble`` CLI tool provides the following options related to disassemble multiple images to small tiles of tile shape

* ``--input_dir``: Directory containing mosaiced images to split

* ``--fmt``: Format of input images in input_dir

* ``--tile_size``: Size of each tile in the split mosaic

* ``--output_dir``: Directory containing all the disassembled images
