import os
import cohere
import numpy as np
import gradio as gr
import json
import getpass
import chromadb
from langchain_core.runnables import RunnablePassthrough
from langchain import PromptTemplate, LLMChain
from pymilvus import MilvusClient
import google.generativeai as genai


def format_the_contexts(res, db_type):
    if db_type == "milvus":
        results = res[0]
        Entity_list = [json.dumps(result["entity"]) for result in results]
        return "\n".join(Entity_list)
    elif db_type == "chromadb":
        Entity_list = [json.dumps(result["entity"]) for result in res]
        return "\n".join(Entity_list)


def set_env_variables(env_vars, condition):
    """
    Sets environment variables from a dictionary based on a condition.

    Args:
        env_vars (dict): A dictionary where keys are environment variable names and values are the values to set.
        condition (str): The condition to check before setting the environment variable.
    """
    for key, value in env_vars.items():
        if key.startswith(condition):
            if value is None:
                value = getpass.getpass(f"Provide your {key}: ")
            os.environ[key] = value


def configure_model(model_type):
    """
    Configures and returns a language model based on the specified model type.

    Args:
        model_type (str): The type of model to configure. Supported values are 'openai', 'gemini', and 'cohere'.

    Returns:
        llm: The configured language model.

    Raises:
        ValueError: If an unsupported model type is provided.
    """
    env_vars = {
        "OPENAI_API_VERSION": "2023-12-01-preview",
        "AZURE_OPENAI_ENDPOINT": "https://gpt4-azureopenai-poc-eastus2.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
        "GEMINI_EMBEDDING_KEY": os.getenv("GEMINI_EMBEDDING_KEY"),
    }

    if model_type == "openai":
        set_env_variables(env_vars, "AZURE_OPENAI")
        from langchain_openai import AzureOpenAI
        import openai

        openai.base_url = os.environ["AZURE_OPENAI_ENDPOINT"]
        llm = AzureOpenAI(deployment_name="gpt-35-turbo-16k")
    elif model_type == "gemini":
        set_env_variables(env_vars, "GOOGLE")
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    elif model_type == "cohere":
        set_env_variables(env_vars, "COHERE")
        from langchain.llms import Cohere

        llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"])
    elif model_type == "cohere-embedding":
        set_env_variables(env_vars, "COHERE")
    elif model_type == "gemini-embedding":
        set_env_variables(env_vars, "GEMINI_EMBEDDING")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if model_type not in ["cohere-embedding", "gemini-embedding"]:
        return llm


def get_embeddings(embedding_type, data):
    """
    Get embeddings for the given data using the specified embedding type.

    Parameters:
    - embedding_type (str): The type of embedding to use ('cohere' or 'genai').
    - data (list of str): The data to be embedded.

    Returns:
    - embeddings (list of list of float): The embeddings for the data.
    """
    if embedding_type == "cohere-embedding":
        configure_model("cohere-embedding")
        cohere_api_key = os.getenv("COHERE_API_KEY")
        co = cohere.Client(cohere_api_key)
        embeddings = co.embed(
            texts=data, input_type="search_query", model="embed-english-v3.0"
        ).embeddings

    elif embedding_type == "gemini-embedding":
        configure_model("gemini-embedding")
        genai_api_key = os.getenv("GEMINI_EMBEDDING_KEY")
        genai.configure(api_key=genai_api_key)
        embeddings = genai.embed_content(
            model="models/text-embedding-004", content=data
        )["embedding"]
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

    return embeddings


def embededing_data(embedding_type):
    # Prepare the data
    docs = [
        {
            "text": "CREATE TABLE IF NOT EXISTS Users (id INT PRIMARY KEY AUTO_INCREMENT,name VARCHAR(50) NOT NULL,email VARCHAR(100) NOT NULL,password VARCHAR(255) NOT NULL,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP)",
            "metadata": {
                "description": "Stores Users information including name, email, and password.",
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Products (id INT PRIMARY KEY AUTO_INCREMENT, name VARCHAR(100) NOT NULL, description TEXT, price DECIMAL(10, 2) NOT NULL, category_id INT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP)",
            "metadata": {
                "description": "Stores product information including name, description, price, and category."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Categories (id INT PRIMARY KEY AUTO_INCREMENT, name VARCHAR(100) NOT NULL, description TEXT)",
            "metadata": {"description": "Stores category information for products."},
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Orders (id INT PRIMARY KEY AUTO_INCREMENT, user_id INT, total_price DECIMAL(10, 2) NOT NULL, status VARCHAR(50) NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP)",
            "metadata": {
                "description": "Stores order information including user ID, total price, and status."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Roles (id INT PRIMARY KEY AUTO_INCREMENT, name VARCHAR(50) NOT NULL, description TEXT)",
            "metadata": {"description": "Stores role information for users."},
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Person (PersonId INT IDENTITY PRIMARY KEY, FirstName NVARCHAR(128) NOT NULL, MiddleInitial NVARCHAR(10), LastName NVARCHAR(128) NOT NULL, DateOfBirth DATE NOT NULL)",
            "metadata": {
                "description": "Stores personal information including first name, middle initial, last name, and date of birth. Each person can be associated with multiple students."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Course (CourseId INT IDENTITY PRIMARY KEY, Name NVARCHAR(50) NOT NULL, Teacher NVARCHAR(256) NOT NULL)",
            "metadata": {
                "description": "Stores course information including name and teacher. Each course can have multiple students enrolled."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Credit (StudentId INT IDENTITY PRIMARY KEY, Grade DECIMAL(5,2) CHECK (Grade <= 100.00), Attempt TINYINT, CONSTRAINT [UQ_studentgrades] UNIQUE CLUSTERED(StudentId, Grade, Attempt))",
            "metadata": {
                "description": "Stores credit information including student ID, grade, and attempt. Each credit record is associated with a student.",
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Student (StudentId INT, PersonId INT, CourseId INT, Email NVARCHAR(256))",
            "metadata": {
                "description": "Stores student information including student ID, person ID, course ID, and email. Each student is associated with a person, a course, and can have multiple credit records."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Permissions (id INT PRIMARY KEY AUTO_INCREMENT, name VARCHAR(50) NOT NULL, description TEXT)",
            "metadata": {
                "description": "Stores permission information including name and description."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS UserRoles (user_id INT, role_id INT, PRIMARY KEY (user_id, role_id))",
            "metadata": {
                "description": "Stores user roles including user ID and role ID."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS OrderItems (id INT PRIMARY KEY AUTO_INCREMENT, order_id INT, product_id INT, quantity INT NOT NULL, price DECIMAL(10, 2) NOT NULL)",
            "metadata": {
                "description": "Stores order items including order ID, product ID, quantity, and price."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Payments (id INT PRIMARY KEY AUTO_INCREMENT, order_id INT, amount DECIMAL(10, 2) NOT NULL, payment_method VARCHAR(50) NOT NULL, status VARCHAR(50) NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "metadata": {
                "description": "Stores payment information including order ID, order amount, payment method, and status."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Shipments (id INT PRIMARY KEY AUTO_INCREMENT, order_id INT, shipment_date TIMESTAMP, status VARCHAR(50) NOT NULL)",
            "metadata": {
                "description": "Stores shipment information including order ID, shipment date, and status."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Reviews (id INT PRIMARY KEY AUTO_INCREMENT, product_id INT, user_id INT, rating INT NOT NULL, comment TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "metadata": {
                "description": "Stores review information including product ID, user ID, rating, and comment."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Addresses (id INT PRIMARY KEY AUTO_INCREMENT, user_id INT, address_line1 VARCHAR(255) NOT NULL, address_line2 VARCHAR(255), city VARCHAR(100) NOT NULL, state VARCHAR(100) NOT NULL, postal_code VARCHAR(20) NOT NULL, country VARCHAR(100) NOT NULL)",
            "metadata": {
                "description": "Stores address information including user ID, address lines, city, state, postal code, and country."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Carts (id INT PRIMARY KEY AUTO_INCREMENT, user_id INT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP)",
            "metadata": {
                "description": "Stores cart information including user ID and timestamps."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS CartItems (id INT PRIMARY KEY AUTO_INCREMENT, cart_id INT, product_id INT, quantity INT NOT NULL)",
            "metadata": {
                "description": "Stores cart items including cart ID, product ID, and quantity."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Wishlists (id INT PRIMARY KEY AUTO_INCREMENT, user_id INT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "metadata": {
                "description": "Stores wishlist information including user ID and timestamp."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Notifications (id INT PRIMARY KEY AUTO_INCREMENT, user_id INT, message TEXT NOT NULL, read BOOLEAN DEFAULT FALSE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "metadata": {
                "description": "Stores notifications including user ID, message, read status, and timestamp."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Messages (id INT PRIMARY KEY AUTO_INCREMENT, sender_id INT, receiver_id INT, message TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "metadata": {
                "description": "Stores messages including sender ID, receiver ID, message, and timestamp."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Settings (id INT PRIMARY KEY AUTO_INCREMENT, user_id INT, setting_key VARCHAR(100) NOT NULL, setting_value TEXT)",
            "metadata": {
                "description": "Stores settings including user ID, setting key, and setting value."
            },
        },
        {
            "text": "CREATE TABLE IF NOT EXISTS Logs (id INT PRIMARY KEY AUTO_INCREMENT, user_id INT, action TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "metadata": {
                "description": "Stores logs including user ID, action, and timestamp."
            },
        },
    ]
    docs_str = [str(doc) for doc in docs]
    doc_emb = get_embeddings(embedding_type, docs_str)
    return doc_emb, docs


def vector_db(db_type, embedding_type):
    embeds, docs = embededing_data(embedding_type)

    if db_type == "milvus":
        client = MilvusClient("milvus.db")
        # Create Milvus db collection
        dimension = 1024 if embedding_type == "cohere-embedding" else 768
        if client.has_collection(collection_name="demo_collection"):
            client.drop_collection(collection_name="demo_collection")
        client.create_collection(
            collection_name="demo_collection",
            dimension=dimension,
        )

        data = [
            {
                "id": i,
                "vector": embeds[i],
                "text": docs[i],
                "subject": "DDL statement for Snowflake table creation",
            }
            for i in range(len(embeds))
        ]
        print("Data has", len(data), "entities, each with fields: ", data[0].keys())
        print("Vector dim:", len(data[0]["vector"]))
        print("data", data)
        print("typeee", type(data))
        # Load the data into Vector DB
        res = client.insert(collection_name="demo_collection", data=data)
        print(res)

    elif db_type == "chromadb":
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(name="my_collection")

        # Convert documents to strings
        docs_str = [str(doc) for doc in docs]

        # Define entities for ChromaDB schema
        data = [
            {
                "id": str(i),
                "vector": embeds[i],
                "text": docs_str[i],
                "subject": "DDL statement for Snowflake table creation",
            }
            for i in range(len(embeds))
        ]

        # Extract ids and documents for upsert
        ids = [item["id"] for item in data]
        documents = [item["text"] for item in data]  # Only the text data

        # Upsert documents into the collection
        collection.upsert(documents=documents, ids=ids)


def get_the_context_from_vectorDB(message, db_type, embedding_type):
    query_vector = get_embeddings(embedding_type, [message])

    if db_type == "milvus":
        client = MilvusClient("milvus.db")
        res = client.search(
            collection_name="demo_collection",
            data=query_vector,
            limit=4,
            output_fields=["text", "subject"],
        )
        print(res)
        return format_the_contexts(res, "milvus")

    elif db_type == "chromadb":
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection("my_collection")
        res = collection.query(query_texts=str(query_vector), n_results=3)

        formatted_res = [
            {
                "id": res["ids"][0][i],
                "distance": res["distances"][0][i],
                "entity": {
                    "text": res["documents"][0][i],
                    "subject": "DDL statement for Snowflake table creation",
                },
            }
            for i in range(len(res["documents"][0]))
        ]
        print(formatted_res)
        return format_the_contexts(formatted_res, "chromadb")


def generate_response(context, query):
    input1 = """\

    Context information is below.

    ---------------------

    {my_context}

    ---------------------

    Given the context information and not prior knowledge, answer the query.
    You have to form a SQL for the query of SQLite db.
    The context is a list of DDL statements for the Tables in a Snowflake database.
    Understand the relation among the columns of the tables by analysing the context.
    Before generating the query, check the data types of the columns in the DDL statements very well.
    The data types of the arguments in the query and data types of columns in the Tables should match, always.
    Generate the SQL of SQLite database not Mysql based on the relation.

    Query: {my_query}

    Answer: \

    """
    prompt = PromptTemplate(template=input1, input_variables=["my_context", "my_query"])

    llm = configure_model("cohere")
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    response = llm_chain.invoke({"my_context": context, "my_query": query})

    return response


# Initialize vector database
vector_db("milvus", "gemini-embedding")


# start
test_context = get_the_context_from_vectorDB(
    "Can you list all students along with the courses they are enrolled in? I need the student's name and the course name.",
    "milvus",
    "gemini-embedding",
)
print(test_context)
test_response = generate_response(
    test_context,
    "Can you list all students along with the courses they are enrolled in? I need the student's name and the course name.",
)
sql = test_response["text"]
print(sql)


# def predict(message, history):
#     context = get_the_context_from_vectorDB(message, "milvus")
#     response = generate_response(context, message)
#     sql = response["text"]
#     return f"What I think the SQL should like is '{sql}'"


# demo = gr.Interface(
#     fn=predict,
#     inputs=gr.Textbox(label="Query", placeholder="Enter your query here"),
#     outputs=[gr.Textbox(label="SQL for you")],
#     title="SQL generator",
#     theme=gr.themes.GoogleFont,  # <-- Prebuilt gr.themes.Base() passed to theme= parameter
# )

# gr.ChatInterface(predict).launch()
