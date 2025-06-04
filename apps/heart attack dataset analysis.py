import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import marimo as mo

    import polars as pl
    import polars.selectors as cs

    import altair as alt
    return alt, mo, pathlib, pl


@app.cell
def _(mo):
    mo.md(r"""# Analyzing Heart Attack Dataset""")
    return


@app.cell
def _(pathlib, pl):
    DATA_DIR = pathlib.Path('../data')
    DATA = DATA_DIR / 'Heart Attack Dataset' / 'Medicaldataset.csv'

    df = pl.read_csv(DATA, infer_schema_length=1000)
    print(df)

    # Transform the data so that Gender is male or female (instead of 0 or 1)
    df = df.with_columns([
        pl.col("Gender").cast(pl.String),
    ])
    df = df.with_columns(pl.col('Gender').replace({'1': 'male', '0': 'female'}))
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""# The Dataset""")
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(r"""## Exploratory Data Analysis""")
    return


@app.cell
def _(mo):
    bp_type = mo.ui.dropdown(
        options=['Systolic', 'Diastolic'],
        value='Systolic',
        label='Blood Pressure Type',
        searchable=True
    )
    bp_type
    return (bp_type,)


@app.cell
def _(alt, bp_type, df):
    alt.Chart(df).mark_boxplot(extent=0.5, size=50).encode(
        alt.X(f'{bp_type.value} blood pressure:Q').scale(zero=False),
        alt.Y('Gender'),
        alt.Color('Result'),
    ).properties(
        width=800,
        height=350,
    ).configure_axis(
        labelFontSize=16,
        titleFontSize=16,
    ).configure_legend(
        titleFontSize=18,
        labelFontSize=15
    )
    return


@app.cell
def _(df, mo, pl):
    total = df.height

    total_female = df.filter(pl.col('Gender') == 'female').height
    positive_female = df.filter(
        (pl.col('Gender') == 'female')
        &
        (pl.col('Result') == 'positive')
    ).height

    total_male = df.filter(pl.col('Gender') == 'male').height
    positive_male = df.filter(
        (pl.col('Gender') == 'male')
        &
        (pl.col('Result') == 'positive')
    ).height

    mo.md(
        f'''
        Out of {total_female} female, {positive_female} females was tested positive
        which is {round(100 * positive_female / total_female, 2)}% of the total female
        number of the dataset and {round(100 * positive_female / total, 2)}% of the
        total person number.

        Also, out of {total_male} males, {positive_male} males was tested positive
        which is {round(100 * positive_male / total_male, 2)}% of the total male
        number of the dataset and {round(100 * positive_male / total, 2)}% of the
        total person number.
        '''
    )
    return


if __name__ == "__main__":
    app.run()
