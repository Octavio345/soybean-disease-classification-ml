import os
import json
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision


print("\nVerificando GPU...")
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("GPU detectada:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

mixed_precision.set_global_policy("mixed_float16")


IMG_SIZE = 300
BATCH_SIZE = 32

EPOCHS_HEAD = 20
EPOCHS_FINE = 20

DATASET_PATH = "dataset"

BEST_WEIGHTS_PATH    = "models/melhor_pesos.weights.h5"
FINAL_MODEL_PATH_TF  = "models/modelo_ml_savedmodel"   
FINAL_WEIGHTS_PATH   = "models/modelo_ml_pesos_finais.weights.h5"  


def carregar_dados():

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_data = test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_data, test_data

def criar_modelo(num_classes):

    base_model = EfficientNetB3(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model

def treinar(model, base_model, train_data, test_data):

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3
    )

    checkpoint = ModelCheckpoint(
        BEST_WEIGHTS_PATH,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    print("\nFASE 1 — Treinando somente o topo (base congelada)")
    model.fit(
        train_data,
        epochs=EPOCHS_HEAD,
        validation_data=test_data,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    print("\nFASE 2 — Fine-tuning (últimas 100 camadas liberadas)")

    base_model.trainable = True

    for layer in base_model.layers[:-100]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_data,
        epochs=EPOCHS_FINE,
        validation_data=test_data,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

def salvar_modelo(num_classes):

    print("salvando modelo final")


    print("\n[1/4] Revertendo política para float32...")
    mixed_precision.set_global_policy("float32")


    print("[2/4] Recriando modelo limpo em float32...")
    modelo_limpo, _ = criar_modelo(num_classes)

  
    print("[3/4] Carregando melhores pesos...")
    modelo_limpo.load_weights(BEST_WEIGHTS_PATH)


    print(f"[4/4] Salvando pesos finais: {FINAL_WEIGHTS_PATH}")
    modelo_limpo.save_weights(FINAL_WEIGHTS_PATH)
    print(f"Pesos finais salvos!")

    print(f"[4/4] Salvando modelo completo (SavedModel): {FINAL_MODEL_PATH_TF}")
    tf.saved_model.save(modelo_limpo, FINAL_MODEL_PATH_TF)
    print(f"SavedModel salvo!")

    print("\nModelo salvo com sucesso!")
    print(f"   → Pesos: {FINAL_WEIGHTS_PATH}")
    print(f"   → Modelo completo: {FINAL_MODEL_PATH_TF}/")
    print(f"\n   Para usar na API:")
    print(f"   model = tf.keras.models.load_model('{FINAL_MODEL_PATH_TF}')")


if __name__ == "__main__":

    os.makedirs("models", exist_ok=True)

    train_data, test_data = carregar_dados()

    classes = list(train_data.class_indices.keys())
    print("\nClasses detectadas:", classes)
    print("Total de classes:", len(classes))

    model, base_model = criar_modelo(len(classes))
    treinar(model, base_model, train_data, test_data)

    salvar_modelo(len(classes))

    with open("models/classes.json", "w") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print("treino finalizado")