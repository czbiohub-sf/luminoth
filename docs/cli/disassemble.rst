.. _cli/disassemble:

Disassemble one large image to small images
============================================

Assuming you have want to split large image into small image::


  $ lumi disassemble --input_img mosaiced.png --tile_size 40, 50 --output_dir split/

The ``lumi disassemble`` CLI tool provides the following options related to disassemble an image of tile shape

* ``--input_img``: Path of the image to split

* ``--tile_size``: Size of each tile in the split mosaic

* ``--output_dir``: Directory containing all the disassembled images
