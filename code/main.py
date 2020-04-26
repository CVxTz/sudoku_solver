import sys

sys.path.append("..")

from utils import gen
from models import get_model

if __name__ == "__main__":
    model = get_model()

    model.load_weights("model.h5", by_name=True)
