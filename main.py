# import json

async def app(scope, receive, send):
    assert scope['type'] == 'http'

    print('scope : ', scope)
    # print(json.dumps(scope.decode("utf-8"), indent=4))

    await send({
        'type': 'http.response.start',
        'status': 200,
        'headers': [
            [b'content-type', b'text/plain'],
        ],
    })
    await send({
        'type': 'http.response.body',
        'body': b'Hello, world!',
    })





# how to run
# uvicorn main:app