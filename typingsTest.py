from typing import Annotated

# Example of using Annotated
def process_data(data: Annotated[int, "This should be a positive integer"]) -> int:
    return data * 2


print(process_data(1))