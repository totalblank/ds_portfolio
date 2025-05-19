import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import sklearn
    import sklearn.datasets
    import sklearn.manifold
    import pandas as pd

    import sys

    if "pyodide" in sys.modules:
        import micropip
        await micropip.install("altair")

    import altair as alt
    return alt, mo, pd, sklearn


@app.cell
def _(mo):
    mo.md(
        "# Embedding Visualizer"
    )
    return


@app.cell
def _(pd, sklearn):
    raw_digits, raw_labels = sklearn.datasets.load_digits(return_X_y=True)

    X_embedded = sklearn.decomposition.PCA(
        n_components=2, whiten=True
    ).fit_transform(raw_digits)

    embedding = pd.DataFrame(
        {"x": X_embedded[:, 0], "y": X_embedded[:, 1], "digit": raw_labels}
    ).reset_index()
    return (embedding,)


@app.cell
def _(alt, embedding, mo):
    def scatter(df):
        return (alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("x:Q").scale(domain=(-2.5, 2.5)),
            y=alt.Y("y:Q").scale(domain=(-2.5, 2.5)),
            color=alt.Color("digit:N"),
        ).properties(width=500, height=500))

    chart = mo.ui.altair_chart(scatter(embedding))
    chart
    return


if __name__ == "__main__":
    app.run()
