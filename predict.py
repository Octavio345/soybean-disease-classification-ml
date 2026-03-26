import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input  

# Carregar modelo (SavedModel — mesmo formato salvo no treino)
model = tf.keras.models.load_model("models/modelo_ml_savedmodel")

# Carregar classes
with open("models/classes.json", "r") as f:
    classes = json.load(f)


def get_model_input_size(loaded_model):
    """Extrai dinamicamente o tamanho esperado pelo modelo."""
    shape = loaded_model.input_shape
    if isinstance(shape, list):
        shape = shape[0]

    height, width = shape[1], shape[2]
    if height is None or width is None:
        raise ValueError(f"Input shape inválido no modelo: {shape}")

    return int(height), int(width)


def predict_image(img_path, threshold=0.60, margin_threshold=0.15):
    img_h, img_w = get_model_input_size(model)

    img = cv2.imread(img_path)

    if img is None:
        print("Erro: imagem não encontrada.")
        print("Caminho recebido:", img_path)
        return

    # BGR -> RGB (OpenCV carrega em BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (img_w, img_h))
    img = preprocess_input(img)          
    img = np.expand_dims(img, axis=0)    

    prediction = model.predict(img, verbose=0)[0]

    print(f"Tamanho de entrada usado: {img_h}x{img_w}")
    print("Soma das probabilidades:", np.sum(prediction))

    sorted_idx = np.argsort(prediction)[::-1]
    idx        = int(sorted_idx[0])
    second_idx = int(sorted_idx[1])
    confidence  = float(prediction[idx])
    second_conf = float(prediction[second_idx])
    margin      = confidence - second_conf

    print("\nProbabilidades:")
    for i, c in enumerate(classes):
        print(f"  {c}: {prediction[i]*100:.2f}%")

    print("\nTop-2:")
    print(f"  1) {classes[idx]}: {confidence*100:.2f}%")
    print(f"  2) {classes[second_idx]}: {second_conf*100:.2f}%")
    print(f"  Margem entre 1º e 2º: {margin*100:.2f} pp")

    print("\nResultado Final:")
    if confidence < threshold or margin < margin_threshold:
        print(" Diagnóstico inconclusivo")
    else:
        print(f"  {classes[idx].replace('_', ' ').upper()} ({confidence*100:.2f}%)")

# Coloque a sua imagem para testar
if __name__ == "__main__":
    predict_image("teste.jpg")