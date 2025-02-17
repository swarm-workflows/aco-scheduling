import json
import numpy as np
from typing import Any

class Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (np.float, np.float32)):
            return float(o)
        return super().default(o)

def store(fn, obj):
    with open(fn, 'w') as f:
        json.dump(obj, f, cls=Encoder)


