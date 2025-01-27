"""Simple command line utility called `lumi`.

The cli is composed of subcommands that are able to handle different tasks
needed for training and using deep learning models.

It's base subcommands are:
    train: For training locally.
    cloud: For traning and monitoring in the cloud.
    dataset: For modifying and transforming datasets.
"""

import click

from luminoth.confusion_matrix import confusion_matrix
from luminoth.eval import eval
from luminoth.predict import predict
from luminoth.tools import checkpoint, cloud, dataset, server
from luminoth.train import train
from luminoth.utils import (
    split_train_val,
    mosaic,
    disassemble,
    overlay_bbs,
    data_aug_demo,
)


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


cli.add_command(checkpoint)
cli.add_command(cloud)
cli.add_command(confusion_matrix)
cli.add_command(dataset)
cli.add_command(eval)
cli.add_command(predict)
cli.add_command(server)
cli.add_command(train)
cli.add_command(split_train_val)
cli.add_command(mosaic)
cli.add_command(disassemble)
cli.add_command(overlay_bbs)
cli.add_command(data_aug_demo)
