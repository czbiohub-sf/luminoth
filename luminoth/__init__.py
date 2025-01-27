__version__ = "0.2.4dev0"

__title__ = "Luminoth"
__description__ = "Computer vision toolkit based on TensorFlow"
__uri__ = "https://github.com/czbiohub/luminoth-uv-imaging/"
__doc__ = __description__ + " <" + __uri__ + ">"

__author__ = "Tryolabs/Pranathi Vemuri"
__email__ = "pranathi93.vemuri@gmail.com"

__license__ = "BSD 3-Clause License"
__copyright__ = "Copyright (c) 2018 Tryolabs S.A."

__min_tf_version__ = "1.5"

import sys

# Check for a current TensorFlow installation.
try:
    import tensorflow  # noqa: F401
except ImportError:
    import sys

    sys.exit(
        """Luminoth requires a TensorFlow >= {} installation.Depending on your use case, you should install either `tensorflow` or
             `tensorflow-gpu` packages manually or via PyPI.""".format(
            __min_tf_version__
        )
    )


# Import functions that are part of Luminoth's public interface.
from luminoth.cli import cli  # noqa
from luminoth.io import read_image  # noqa
from luminoth.tasks import Detector  # noqa
from luminoth.utils.vis import vis_objects  # noqa
