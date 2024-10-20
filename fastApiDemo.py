from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"
    

class Image(BaseModel):
    url: str
    name: str

class Item(BaseModel):
    name:str
    description: str|None = None
    price: float
    tax: float | None = None
    tags: set[str] = set()
    image: Image | None = None
    

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet": 
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}


# http://127.0.0.1:8000/items2/?skip=0&limit=10
@app.get("/items2/")
async def read_item2(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]



@app.post("/items/")
async def create_item(item: Item):
    return item

"""
{
    "name": "Foo",
    "description": "An optional description",
    "price": 45.2,
    "tax": 3.5
} 
"""









# swagger 
# http://127.0.0.1:8000/docs

# http://127.0.0.1:8000/redoc