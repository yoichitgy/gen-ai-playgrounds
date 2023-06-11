import json
import os
import sys
from tempfile import NamedTemporaryFile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from google.cloud import storage

BUCKET = "cloud-samples-data"
PREFIX = "ai-platform/flowers/"

tf.keras.utils.disable_interactive_logging()
model = tf.keras.applications.EfficientNetB0(include_top=False, pooling="avg")


def blob_to_embedding(blob: storage.Blob) -> list[float]:
    with NamedTemporaryFile(prefix="flowers") as temp:
        blob.download_to_filename(temp.name)
        raw = tf.io.read_file(temp.name)

    image = tf.image.decode_jpeg(raw, channels=3)
    prediction = model.predict(np.array([image.numpy()]))
    return prediction[0].tolist()


def generate_and_upload_embeddings(flower: str, destination_root: str) -> None:
    print(f"Generating and uploading embeddings for {flower}...")

    client = storage.Client(project="gs-matching-engine")
    datapoints = []
    blobs = list(client.list_blobs(BUCKET, prefix=f"{PREFIX}{flower}/"))
    for i, blob in enumerate(blobs, 1):
        print(f"Processing {i}/{len(blobs)}: {blob.name}")
        datapoints.append({
            "id": blob.name,
            "embedding": blob_to_embedding(blob),
        })
    
    dst_bucket_name, dst_base = destination_root[5:].split("/", maxsplit=1)
    dst_bucket = client.bucket(dst_bucket_name)
    dst_blob = dst_bucket.blob(os.path.join(dst_base, f"{flower}.json"))

    with dst_blob.open("w") as f:
        for datapoint in datapoints:
            f.write(json.dumps(datapoint) + "\n")

    print(f"Done! Embeddings for {flower} are available at {dst_blob.public_url}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("""Usage: python generate_and_upload_embeddings.py <flower> <destination_root>
          flower: daisy, dandelion, roses, sunflowers or tulips
          destination_root: gs://<bucket>/<path>""")

    generate_and_upload_embeddings(sys.argv[1], sys.argv[2])
