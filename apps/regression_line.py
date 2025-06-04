import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""# Finding the best fit line""")
    return


@app.cell
async def _():
    import marimo as mo
    import pandas as pd
    import sys
    import random
    import altair_transform

    if "pyodide" in sys.modules:
        import micropip
        await micropip.install("altair")

    import altair as alt
    return alt, altair_transform, mo, pd, random


@app.cell
def _(mo):
    select_dataset = mo.ui.dropdown(
        options=["Iris", "Cars", "Climate"],
        value="Cars",
        label="Choose dataset"
    )
    select_dataset
    return (select_dataset,)


@app.cell
def _(pd, select_dataset):
    dataset_url = f"https://cdn.jsdelivr.net/npm/vega-datasets@latest/data/{select_dataset.value.lower()}.json"

    df = pd.read_json(dataset_url)
    df
    return (df,)


@app.cell
def _(df, mo, random):
    valid_columns_2 = list(filter(lambda column: df[column].dtype != 'object', list(df.columns)))

    dropdown_2 = mo.ui.dropdown(
        options=valid_columns_2,
        label="Choose x",

        # choose a random column
        value = random.choice(valid_columns_2)
    )
    return dropdown_2, valid_columns_2


@app.cell
def _(dropdown_2, mo, random, valid_columns_2):
    valid_columns_3 = list(filter(lambda x: x, [x if not x == dropdown_2.value else None for x in valid_columns_2]))
    dropdown_3 = mo.ui.dropdown(
        options=valid_columns_3,
        label="Choose y",
        # choose a random column
        value = random.choice(valid_columns_3)
    )
    return (dropdown_3,)


@app.cell
def _(mo):
    mo.md(r"""## Choose columns""")
    return


@app.cell
def _(dropdown_2, dropdown_3):
    dropdown_2, dropdown_3
    return


@app.cell
def _(mo):
    mo.md(r"""## Scatter plot""")
    return


@app.cell
def _(alt, df, dropdown_2, dropdown_3, random, valid_columns_2):
    chart = alt.Chart(df).mark_circle(size=100).encode(
        x=dropdown_2.value,
        y=dropdown_3.value,
        color=f'{random.choice(valid_columns_2)}:N',
        tooltip=valid_columns_2
    )

    line = chart.transform_regression(
        dropdown_2.value,
        dropdown_3.value
    ).mark_line()

    params = alt.Chart(df).transform_regression(
        dropdown_2.value, dropdown_3.value, params=True
    ).mark_text(align='left').encode(
        x=alt.value(20),  # pixels from left
        y=alt.value(20),  # pixels from top
        text="rSquared:N"
    )

    (chart + line + params).interactive()
    return (line,)


@app.cell
def _(altair_transform, line):
    print(altair_transform.extract_data(line))
    return


if __name__ == "__main__":
    app.run()
