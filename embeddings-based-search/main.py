import typer

app = typer.Typer()
EMBEDDINGS_NAME = "winter_olympics_2022.csv"
DATAFRAME_NAME = "winter_olympics_2022.pkl"


@app.command()
def prep_dataframe():
    import ast
    import os

    import pandas as pd

    embeddings_url = "https://cdn.openai.com/API/examples/data/" + EMBEDDINGS_NAME

    # Check the file exists
    if os.path.exists(EMBEDDINGS_NAME):
        print("Using the existing embedding file.")
    else:
        print("Downloading the embedding file...")
        download_file(embeddings_url, EMBEDDINGS_NAME)
        print("Downloaded the embedding file.")

    print("Reading the embedding file...")
    df = pd.read_csv(EMBEDDINGS_NAME)
    print("Evaluting the embeddings...")
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    df.to_pickle("winter_olympics_2022.pkl")


def download_file(url: str, dest: str):
    import requests

    r = requests.get(url, allow_redirects=True)
    if r.status_code != 200:
        raise Exception(f"Could not download file {url}")

    with open(dest, "wb") as f:
        f.write(r.content)


@app.command()
def read_dataframe():
    import os

    import pandas as pd

    if not os.path.exists(DATAFRAME_NAME):
        prep_dataframe()

    df = pd.read_pickle(DATAFRAME_NAME)
    print(df)


if __name__ == "__main__":
    app()
