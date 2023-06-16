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
PINECONE_INDEX = "winter-olympics-2022"


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
    query: str, top_n: int = 100
) -> tuple[list[str], list[float]]:
    def relatedness_fn(x, y):
        return 1 - spatial.distance.cosine(x, y)

    df = read_dataframe()
    query_embedding = embedding_from_query(query)
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for _, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)

    return strings[:top_n], relatednesses[:top_n]


def embedding_from_query(query: str) -> list[float]:
    EMBEDDING_MODEL = "text-embedding-ada-002"
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return query_embedding_response["data"][0]["embedding"]


def read_dataframe() -> pd.DataFrame:
    import os

    if not os.path.exists(DATAFRAME_NAME):
        prep_dataframe()
    df = pd.read_pickle(DATAFRAME_NAME)
    return df


def num_tokens(text: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(query: str, model: str, token_budget: int, use_pinecone: bool) -> str:
    if use_pinecone:
        strings, _ = strings_queried_to_pinecone(query)
    else:
        strings, _ = strings_ranked_by_relatedness(query)
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
    use_pinecone: bool = False,
):
    message = query_message(query, model, token_budget, use_pinecone)
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


@app.command()
def create_pinecone_index():
    import pinecone

    init_pinecone()
    pinecone.create_index(PINECONE_INDEX, dimension=1536, metric="cosine")


@app.command()
def upsert_pinecone():
    import pinecone

    init_pinecone()
    index = pinecone.Index(PINECONE_INDEX)

    df = read_dataframe()
    for i, row in df.iterrows():
        print(f"Upserting {i}...")
        vector = (f"id_{i}", row["embedding"], {"text": row["text"]})
        index.upsert([vector])


def strings_queried_to_pinecone(
    query: str, top_n: int = 100
) -> tuple[list[str], list[float]]:
    import pinecone

    init_pinecone()
    index = pinecone.Index(PINECONE_INDEX)
    query_embedding = embedding_from_query(query)

    resp = index.query(query_embedding, top_k=top_n, include_metadata=True)

    strings_and_scores = [(match["metadata"]["text"], match["score"]) for match in resp["matches"]]
    strings, scores = zip(*strings_and_scores)
    return strings, scores


def init_pinecone():
    import os

    import pinecone

    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)


if __name__ == "__main__":
    app()
