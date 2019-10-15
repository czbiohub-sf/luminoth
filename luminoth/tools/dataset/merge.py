import click
import pandas as pd
import tensorflow as tf


@click.command()
@click.argument('src', nargs=-1)
@click.argument('dst', nargs=1)
@click.option('--type', type=str, default="tf", help='Type of datasets to merge.')
@click.option('--debug', is_flag=True, help='Set level logging to DEBUG.')
def merge(src, dst, debug):
    """
    Merges existing datasets into a single one.
    """

    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info('Saving records to "{}"'.format(dst))

    if type == "tf":
        writer = tf.python_io.TFRecordWriter(dst)

        total_records = 0

        for src_file in src:
            total_src_records = 0
            for record in tf.python_io.tf_record_iterator(src_file):
                writer.write(record)
                total_src_records += 1
                total_records += 1

            tf.logging.info('Saved {} records from "{}"'.format(
                total_src_records, src_file))

        tf.logging.info('Saved {} to "{}"'.format(total_records, dst))

        writer.close()
    elif type == "csv":
        total_records = 0
        dfs = []

        for src_file in src:
            df = pd.read_csv(src_file, sep=",")
            total_src_records = len(df)
            tf.logging.info('Saved {} csv records from "{}"'.format(
                total_src_records, src_file))
            dfs.append(df)

        merged_df = pd.concat(dfs)

        tf.logging.info('Saved {} to "{}"'.format(total_records, dst))

        merged_df.to_csv(dst)
