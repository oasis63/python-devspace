from pymilvus import MilvusClient, DataType
import random
import string

# client = MilvusClient(
#     uri="http://localhost:19530"  # replace with your own Milvus server address
# )


client = MilvusClient(
    uri="http://192.168.0.103:19530"  # replace with your own Milvus server address
)


def create_milvus_collection(collection_name: string, dimension: int):

    client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=False, enabel_dynamic_field=True)

    schema.add_field(
        field_name="doc_id",
        datatype=DataType.INT64,
        is_primary=True,
        description="document id",
    )
    schema.add_field(
        field_name="doc_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=dimension,
        description="document vector",
    )
    schema.add_field(
        field_name="doc_text",
        datatype=DataType.VARCHAR,
        max_length=65535,
        description="document text",
    )

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="doc_vector",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128},
    )

    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )


def generate_random_data():
    # Generate random data for entities
    num_samples = 10  # Number of sample entries to generate

    sample_data = []

    for i in range(num_samples):
        doc_id = i + 1  # Generate a unique document ID
        doc_vector = [
            random.random() for _ in range(dimension)
        ]  # Generate a random vector of the specified dimension
        doc_text = "".join(
            random.choices(string.ascii_letters, k=50)
        )  # Generate a random string of 50 characters

        entity = {"doc_id": doc_id, "doc_vector": doc_vector, "doc_text": doc_text}

        sample_data.append(entity)

    return sample_data


# start the process here

collection_name = "first_my_collection"
# Define the dimension of the embedding vector
dimension = 384

create_milvus_collection(collection_name, dimension)
entities = generate_random_data()
client.insert(collection_name=collection_name, data=entities)

print("-----------done------------------")
