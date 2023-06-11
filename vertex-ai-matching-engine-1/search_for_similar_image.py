import os
import sys

import numpy as np
from google.cloud import aiplatform_v1beta1 as vertexai
from google.cloud.aiplatform.matching_engine import MatchingEngineIndexEndpoint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.keras.utils.disable_interactive_logging()


def file_to_embedding(model: tf.keras.Model, path: str) -> list[float]:
    raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    prediction = model.predict(np.array([image.numpy()]))
    return prediction[0].tolist()


class Matcher:
    def __init__(self, index_endpoint_name: str, deployed_index_id: str):
        self._index_endpoint_name = index_endpoint_name
        self._deployed_index_id = deployed_index_id
    
        self._client = vertexai.MatchServiceClient(
            client_options={"api_endpoint": self._public_endpoint()}
        )

    def find_neighbors(self, embedding: list[float], neighbor_count: int):
        datapoint = vertexai.IndexDatapoint(
            datapoint_id="dummy-id",
            feature_vector=embedding
        )
        query = vertexai.FindNeighborsRequest.Query(
            datapoint=datapoint,
            neighbor_count=neighbor_count
        )
        request = vertexai.FindNeighborsRequest(
            index_endpoint=self._index_endpoint_name,
            deployed_index_id=self._deployed_index_id,
            queries=[query]
        )
        resp = self._client.find_neighbors(request)
        print(type(resp.nearest_neighbors[0].neighbors))
        return resp.nearest_neighbors[0].neighbors

    def _public_endpoint(self) -> str:
        endpoint = MatchingEngineIndexEndpoint(
            index_endpoint_name=self._index_endpoint_name
        )
        return endpoint.gca_resource.public_endpoint_domain_name


def search_for_similar_image(
    index_endpoint_name: str,
    deployed_index_id: str,
    image_path: str
) -> None:
    print("Loading EfficientNetB0...")
    model = tf.keras.applications.EfficientNetB0(include_top=False, pooling="avg")

    print(f"Started generating embedding for {image_path}...")
    embedding = file_to_embedding(model, image_path)
    print("Done!")

    matcher = Matcher(index_endpoint_name, deployed_index_id)
    neighbors = matcher.find_neighbors(embedding, 10)

    for neighbor in neighbors:
        datapoint_id = neighbor.datapoint.datapoint_id
        distance = neighbor.distance
        print(f"Found {datapoint_id} at distance {distance}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("""Usage: python search_for_similar_image.py <image_path>
          image_path: The path to the image to search for""")
    
    index_endpoint_name = os.environ["INDEX_ENDPOINT_NAME"]
    deployed_index_id = os.environ["DEPLOYED_INDEX_ID"]
    image_path = sys.argv[1]

    search_for_similar_image(
        index_endpoint_name=index_endpoint_name,
        deployed_index_id=deployed_index_id,
        image_path=image_path
    )
