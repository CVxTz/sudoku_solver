import sys

sys.path.append("..")

from solver.utils import gen
from solver.solver_models import get_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

if __name__ == "__main__":
    model = get_model()

    checkpoint = ModelCheckpoint(
        "solver.h5", monitor="acc", verbose=1, save_best_only=True, mode="max",
    )
    reduce = ReduceLROnPlateau(monitor="acc", mode="max", patience=100, min_lr=1e-7)

    try:
        model.load_weights("solver.h5", by_name=True)
    except:
        print("Could not load the model")

    model.fit_generator(
        gen(),
        epochs=20000,
        steps_per_epoch=128,
        callbacks=[checkpoint, reduce],
        validation_data=gen(),
        validation_steps=32,
        use_multiprocessing=True,
        workers=8
    )
