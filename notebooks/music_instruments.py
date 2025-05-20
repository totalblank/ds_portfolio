import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""# Classifying Musical Instruments using CNN""")
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import os
    import numpy as np
    import seaborn as sns
    import altair as alt
    import warnings
    warnings.simplefilter(action='ignore', category=Warning)

    import random
    from glob import glob
    import cv2
    from PIL import Image
    return mo, os


@app.cell
def _(os):
    DATASET_DIR = "../data/music_instruments/"

    # List all the sub directories
    for dirname, _, filenames in os.walk(DATASET_DIR):
        if not dirname == DATASET_DIR:
            print(dirname)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
