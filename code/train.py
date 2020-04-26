import sys

sys.path.append("..")

from utils import gen
from models import get_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

if __name__ == "__main__":
    model = get_model()

    checkpoint = ModelCheckpoint(
        "model.h5", monitor="acc", verbose=1, save_best_only=True, mode="max",
    )
    reduce = ReduceLROnPlateau(monitor="acc", mode="max", patience=30, min_lr=1e-7)
    early = EarlyStopping(monitor="acc", mode="max", patience=100)

    try:
        model.load_weights("model.h5", by_name=True)
    except:
        print("Could not load the model")

    model.fit_generator(
        gen(),
        epochs=20000,
        steps_per_epoch=128,
        callbacks=[checkpoint, reduce, early],
        validation_data=gen(),
        validation_steps=64,
    )
