import ast

import openai
import pandas as pd
import tiktoken
import typer
from scipy import spatial

app = typer.Typer()
EMBEDDINGS_NAME = "winter_olympics_2022.csv"
DATAFRAME_NAME = "winter_olympics_2022.pkl"
GPT_MODEL = "gpt-3.5-turbo"


@app.command()
def prep_dataframe():
    import os
    
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


def strings_ranked_by_relatedness(
    query: str, df: pd.DataFrame, top_n: int = 100
) -> tuple[list[str], list[float]]:
    EMBEDDING_MODEL = "text-embedding-ada-002"

    def relatedness_fn(x, y):
        return 1 - spatial.distance.cosine(x, y)

    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for _, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)

    return strings[:top_n], relatednesses[:top_n]


def read_dataframe() -> pd.DataFrame:
    import os

    if not os.path.exists(DATAFRAME_NAME):
        prep_dataframe()
    df = pd.read_pickle(DATAFRAME_NAME)
    return df


def num_tokens(text: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(query: str, df: pd.DataFrame, model: str, token_budget: int) -> str:
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model) > token_budget:
            break
        message += next_article

    return message + question


@app.command()
def ask(
    query: str,
    model: str = GPT_MODEL,
    temperature: float = 0,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
):
    df = read_dataframe()
    message = query_message(query, df, model, token_budget)
    if print_message:
        print(message)

    messages = [
        {
            "role": "system",
            "content": "You answer questions about the 2022 Winter Olympics.",
        },
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature
    )
    response_message = response["choices"][0]["message"]["content"]
    print(response_message)


if __name__ == "__main__":
    app()
