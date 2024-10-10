import json

def store(fn, obj):
    with open(fn, 'w') as f:
        json.dump(obj, f)


