# 🌱 Classificação de Doenças em Soja — TCC

> Modelo de visão computacional baseado em **EfficientNetB3** para identificação automática de doenças foliares em plantas de soja, desenvolvido como Trabalho de Conclusão de Curso.

---

> 🇺🇸 **English version available below** — [Jump to English](#-soybean-disease-classification--tcc)

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Classes Detectadas](#-classes-detectadas)
- [Arquitetura do Modelo](#-arquitetura-do-modelo)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
  - [Treinamento](#treinamento)
  - [Predição](#predição)
- [Pipeline de Treinamento](#-pipeline-de-treinamento)
- [Arquivos Gerados](#-arquivos-gerados)
- [Parâmetros Configuráveis](#-parâmetros-configuráveis)

---

## 🔍 Visão Geral

Este projeto implementa um classificador de imagens para diagnóstico de doenças em folhas de soja. Utiliza **transfer learning** com EfficientNetB3 pré-treinado no ImageNet, seguido de **fine-tuning** das últimas camadas para adaptação ao domínio agrícola.

O sistema é capaz de distinguir entre plantas saudáveis e três condições patológicas, retornando a classe predita junto com o nível de confiança e uma margem de segurança entre as duas hipóteses mais prováveis.

---

## 🦠 Classes Detectadas

| Classe | Descrição |
|--------|-----------|
| `Ataque_de_largata_Soja` | Danos causados por lagartas nas folhas |
| `Cercospora` | Mancha foliar causada pelo fungo *Cercospora sojina* |
| `Doenca_de_Ferrugem_Soja` | Ferrugem asiática (*Phakopsora pachyrhizi*) |
| `Soja_Saudavel` | Planta sem sinais de doença ou ataque |

---

## 🧠 Arquitetura do Modelo

```
EfficientNetB3 (ImageNet, congelado na fase 1)
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dense(512, relu)  →  Dropout(0.4)
        ↓
Dense(256, relu)  →  Dropout(0.3)
        ↓
Dense(num_classes, softmax, float32)
```

**Estratégia de treinamento em duas fases:**

- **Fase 1 — Head training:** base congelada, apenas as camadas densas são treinadas (`lr = 3e-4`)
- **Fase 2 — Fine-tuning:** últimas 100 camadas da base liberadas, ajuste fino com taxa de aprendizado baixa (`lr = 1e-5`)

**Otimizações:**
- Mixed Precision (`float16`) para acelerar o treino em GPU
- `EarlyStopping` com `patience=5` monitorando `val_loss`
- `ReduceLROnPlateau` com fator `0.3` e `patience=3`
- `ModelCheckpoint` salvando sempre os melhores pesos

---

## 📁 Estrutura do Projeto

```
TCC_VSCODE/
│
├── dataset/
│   ├── train/
│   │   ├── Ataque_de_largata_Soja/
│   │   ├── Cercospora/
│   │   ├── Doenca_de_Ferrugem_Soja/
│   │   └── Soja_Saudavel/
│   └── test/
│       ├── Ataque_de_largata_Soja/
│       ├── Cercospora/
│       ├── Doenca_de_Ferrugem_Soja/
│       └── Soja_Saudavel/
│
├── models/
│   ├── modelo_ml_savedmodel/                  ← Modelo completo (SavedModel)
│   ├── classes.json                           ← Mapeamento índice → classe
│   ├── melhor_pesos.weights.h5                ← Melhores pesos (checkpoint)
│   └── modelo_ml_pesos_finais.weights.h5      ← Pesos finais em float32
│
├── train.py                                   ← Script de treinamento
├── predict.py                                 ← Script de inferência
├── requirements.txt
└── README.md
```

---

## ⚙️ Requisitos

| Pacote | Versão |
|--------|--------|
| `tensorflow` | 2.15.0 |
| `keras` | 2.15.0 |
| `numpy` | 1.26.4 |
| `opencv-python` | 4.9.0.80 |
| `matplotlib` | 3.8.3 |
| `pillow` | 10.2.0 |
| `scikit-learn` | 1.4.1.post1 |
| `h5py` | 3.10.0 |
| `protobuf` | 3.20.3 |

> ⚠️ Use exatamente estas versões para garantir compatibilidade entre TensorFlow, Keras e a serialização dos pesos `.h5`.

---

## 🗂️ Dataset

> ⚠️ **O dataset não está incluído** no repositório devido ao tamanho dos arquivos. O modelo foi treinado em um dataset personalizado contendo imagens de doenças em soja.

Para utilizar este projeto, organize suas próprias imagens na estrutura de pastas descrita em [Estrutura do Projeto](#-estrutura-do-projeto), respeitando uma subpasta por classe dentro de `dataset/train/` e `dataset/test/`.

---

## 🚀 Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/Octavio345/soybean-disease-classification-ml.git
cd tcc-soja

# 2. Crie e ative o ambiente virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Instale as dependências com versões fixas
pip install -r requirements.txt
```

---

## 💻 Como Usar

### Treinamento

Organize seu dataset na estrutura de pastas descrita acima e execute:

```bash
python train.py
```

O script irá:
1. Detectar automaticamente GPU disponível e habilitar mixed precision
2. Carregar e aplicar data augmentation nas imagens de treino
3. Executar as duas fases de treinamento
4. Salvar o modelo final e os pesos em `models/`
5. Gerar o arquivo `models/classes.json` com o mapeamento das classes

**Saída esperada no terminal:**
```
Verificando GPU...
GPU detectada: [PhysicalDevice(name='/physical_device:GPU:0', ...)]

Classes detectadas: ['Ataque_de_largata_Soja', 'Cercospora', ...]
Total de classes: 4

FASE 1 — Treinando somente o topo (base congelada)
...
FASE 2 — Fine-tuning (últimas 100 camadas liberadas)
...
```

---

### Predição

Para classificar uma imagem, edite a última linha do `predict.py` com o caminho desejado e execute:

```bash
python predict.py
```

Ou importe a função diretamente em outro script:

```python
from predict import predict_image

predict_image("caminho/para/imagem.jpg")
```

**Saída esperada:**
```
Tamanho de entrada usado: 300x300
Soma das probabilidades: 1.0

📊 Probabilidades:
  Ataque_de_largata_Soja: 3.12%
  Cercospora: 1.45%
  Doenca_de_Ferrugem_Soja: 92.30%
  Soja_Saudavel: 3.13%

🔎 Top-2:
  1) Doenca_de_Ferrugem_Soja: 92.30%
  2) Soja_Saudavel: 3.13%
  Margem entre 1º e 2º: 89.17 pp

🎯 Resultado Final:
  DOENCA DE FERRUGEM SOJA (92.30%)
```

> **Diagnóstico inconclusivo** é retornado quando a confiança for menor que `60%` **ou** a margem entre 1º e 2º lugar for menor que `15 pontos percentuais`.

---

## 🔄 Pipeline de Treinamento

```
Imagens (dataset/)
       ↓
ImageDataGenerator
  • preprocess_input (EfficientNet)
  • rotation_range=25
  • zoom_range=0.2
  • horizontal_flip=True
       ↓
EfficientNetB3 (300×300×3)
       ↓
  [FASE 1] Head training — 20 épocas máx.
  base_model.trainable = False
  Adam(lr=3e-4)
       ↓
  [FASE 2] Fine-tuning — 20 épocas máx.
  Últimas 100 camadas liberadas
  Adam(lr=1e-5)
       ↓
Salvar como SavedModel + .weights.h5
```

---

## 📦 Arquivos Gerados

| Arquivo | Descrição | Uso |
|---------|-----------|-----|
| `models/modelo_ml_savedmodel/` | Modelo completo em formato Protobuf | Inferência em produção |
| `models/modelo_ml_pesos_finais.weights.h5` | Pesos finais em float32 | Backup / fine-tuning futuro |
| `models/melhor_pesos.weights.h5` | Melhores pesos do checkpoint | Recuperação do melhor estado |
| `models/classes.json` | Lista ordenada das classes | Mapeamento índice → label |

> ⚠️ **Por que SavedModel e não `.h5` ou `.keras`?**  
> O EfficientNetB3 contém EagerTensors internos (médias de normalização do ImageNet) que não são serializáveis para JSON pelo Keras 2.x. O formato SavedModel usa Protobuf binário e contorna esse problema.  
> Para carregar: `model = tf.keras.models.load_model("models/modelo_ml_savedmodel")`

---

## 🔧 Parâmetros Configuráveis

Edite as constantes no topo de `train.py`:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `IMG_SIZE` | `300` | Resolução de entrada (px) |
| `BATCH_SIZE` | `32` | Tamanho do batch |
| `EPOCHS_HEAD` | `20` | Épocas máximas — Fase 1 |
| `EPOCHS_FINE` | `20` | Épocas máximas — Fase 2 |
| `DATASET_PATH` | `"dataset"` | Caminho raiz do dataset |

Edite em `predict.py`:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `threshold` | `0.60` | Confiança mínima para diagnóstico |
| `margin_threshold` | `0.15` | Margem mínima entre 1º e 2º lugar |

---

<br>
<br>

---
---

<br>

# 🌱 Soybean Disease Classification — TCC

> A computer vision model based on **EfficientNetB3** for automatic detection of leaf diseases in soybean plants, developed as a Final Graduation Project (TCC).

---

> 🇧🇷 **Versão em português disponível acima** — [Ir para o Português](#-classificação-de-doenças-em-soja--tcc)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Detected Classes](#-detected-classes)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements-1)
- [Installation](#-installation)
- [How to Use](#-how-to-use)
  - [Training](#training)
  - [Prediction](#prediction)
- [Training Pipeline](#-training-pipeline)
- [Generated Files](#-generated-files)
- [Configurable Parameters](#-configurable-parameters)

---

## 🔍 Overview

This project implements an image classifier for diagnosing diseases in soybean leaves. It uses **transfer learning** with EfficientNetB3 pre-trained on ImageNet, followed by **fine-tuning** of the last layers to adapt to the agricultural domain.

The system can distinguish between healthy plants and three pathological conditions, returning the predicted class along with a confidence score and a safety margin between the two most likely hypotheses.

---

## 🦠 Detected Classes

| Class | Description |
|-------|-------------|
| `Ataque_de_largata_Soja` | Leaf damage caused by caterpillars |
| `Cercospora` | Leaf spot caused by the fungus *Cercospora sojina* |
| `Doenca_de_Ferrugem_Soja` | Asian soybean rust (*Phakopsora pachyrhizi*) |
| `Soja_Saudavel` | Plant with no signs of disease or pest attack |

---

## 🧠 Model Architecture

```
EfficientNetB3 (ImageNet weights, frozen in phase 1)
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dense(512, relu)  →  Dropout(0.4)
        ↓
Dense(256, relu)  →  Dropout(0.3)
        ↓
Dense(num_classes, softmax, float32)
```

**Two-phase training strategy:**

- **Phase 1 — Head training:** base frozen, only dense layers are trained (`lr = 3e-4`)
- **Phase 2 — Fine-tuning:** last 100 base layers unfrozen, fine adjustment with low learning rate (`lr = 1e-5`)

**Optimizations:**
- Mixed Precision (`float16`) to speed up GPU training
- `EarlyStopping` with `patience=5` monitoring `val_loss`
- `ReduceLROnPlateau` with factor `0.3` and `patience=3`
- `ModelCheckpoint` always saving the best weights

---

## 📁 Project Structure

```
TCC_VSCODE/
│
├── dataset/
│   ├── train/
│   │   ├── Ataque_de_largata_Soja/
│   │   ├── Cercospora/
│   │   ├── Doenca_de_Ferrugem_Soja/
│   │   └── Soja_Saudavel/
│   └── test/
│       ├── Ataque_de_largata_Soja/
│       ├── Cercospora/
│       ├── Doenca_de_Ferrugem_Soja/
│       └── Soja_Saudavel/
│
├── models/
│   ├── modelo_ml_savedmodel/                  ← Full model (SavedModel format)
│   ├── classes.json                           ← Index → class mapping
│   ├── melhor_pesos.weights.h5                ← Best weights checkpoint
│   └── modelo_ml_pesos_finais.weights.h5      ← Final weights in float32
│
├── train.py                                   ← Training script
├── predict.py                                 ← Inference script
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

| Package | Version |
|---------|---------|
| `tensorflow` | 2.15.0 |
| `keras` | 2.15.0 |
| `numpy` | 1.26.4 |
| `opencv-python` | 4.9.0.80 |
| `matplotlib` | 3.8.3 |
| `pillow` | 10.2.0 |
| `scikit-learn` | 1.4.1.post1 |
| `h5py` | 3.10.0 |
| `protobuf` | 3.20.3 |

> ⚠️ Use these exact versions to ensure compatibility between TensorFlow, Keras and `.h5` weight serialization.

---

## 🗂️ Dataset

> ⚠️ **Dataset not included** in the repository due to size limitations. The model was trained on a custom dataset containing soybean disease images.

To use this project, organize your own images in the folder structure described in [Project Structure](#-project-structure), with one subfolder per class inside `dataset/train/` and `dataset/test/`.

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Octavio345/soybean-disease-classification-ml.git
cd tcc-soja

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 💻 How to Use

### Training

Organize your dataset in the folder structure described above and run:

```bash
python train.py
```

The script will:
1. Automatically detect available GPU and enable mixed precision
2. Load and apply data augmentation to training images
3. Run both training phases
4. Save the final model and weights to `models/`
5. Generate `models/classes.json` with the class mapping

**Expected terminal output:**
```
Verificando GPU...
GPU detectada: [PhysicalDevice(name='/physical_device:GPU:0', ...)]

Classes detectadas: ['Ataque_de_largata_Soja', 'Cercospora', ...]
Total de classes: 4

FASE 1 — Treinando somente o topo (base congelada)
...
FASE 2 — Fine-tuning (últimas 100 camadas liberadas)
...
```

---

### Prediction

To classify an image, edit the last line of `predict.py` with the desired path and run:

```bash
python predict.py
```

Or import the function directly into another script:

```python
from predict import predict_image

predict_image("path/to/image.jpg")
```

**Expected output:**
```
Tamanho de entrada usado: 300x300
Soma das probabilidades: 1.0

📊 Probabilidades:
  Ataque_de_largata_Soja: 3.12%
  Cercospora: 1.45%
  Doenca_de_Ferrugem_Soja: 92.30%
  Soja_Saudavel: 3.13%

🔎 Top-2:
  1) Doenca_de_Ferrugem_Soja: 92.30%
  2) Soja_Saudavel: 3.13%
  Margem entre 1º e 2º: 89.17 pp

🎯 Resultado Final:
  DOENCA DE FERRUGEM SOJA (92.30%)
```

> **Inconclusive diagnosis** is returned when confidence is below `60%` **or** the margin between 1st and 2nd place is below `15 percentage points`.

---

## 🔄 Training Pipeline

```
Images (dataset/)
       ↓
ImageDataGenerator
  • preprocess_input (EfficientNet)
  • rotation_range=25
  • zoom_range=0.2
  • horizontal_flip=True
       ↓
EfficientNetB3 (300×300×3)
       ↓
  [PHASE 1] Head training — max 20 epochs
  base_model.trainable = False
  Adam(lr=3e-4)
       ↓
  [PHASE 2] Fine-tuning — max 20 epochs
  Last 100 layers unfrozen
  Adam(lr=1e-5)
       ↓
Save as SavedModel + .weights.h5
```

---

## 📦 Generated Files

| File | Description | Usage |
|------|-------------|-------|
| `models/modelo_ml_savedmodel/` | Full model in Protobuf format | Production inference |
| `models/modelo_ml_pesos_finais.weights.h5` | Final weights in float32 | Backup / future fine-tuning |
| `models/melhor_pesos.weights.h5` | Best checkpoint weights | Best state recovery |
| `models/classes.json` | Ordered list of classes | Index → label mapping |

> ⚠️ **Why SavedModel instead of `.h5` or `.keras`?**  
> EfficientNetB3 contains internal EagerTensors (ImageNet normalization means) that cannot be serialized to JSON by Keras 2.x. The SavedModel format uses binary Protobuf and bypasses this issue.  
> To load: `model = tf.keras.models.load_model("models/modelo_ml_savedmodel")`

---

## 🔧 Configurable Parameters

Edit the constants at the top of `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | `300` | Input resolution (px) |
| `BATCH_SIZE` | `32` | Batch size |
| `EPOCHS_HEAD` | `20` | Max epochs — Phase 1 |
| `EPOCHS_FINE` | `20` | Max epochs — Phase 2 |
| `DATASET_PATH` | `"dataset"` | Dataset root path |

Edit in `predict.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.60` | Minimum confidence for diagnosis |
| `margin_threshold` | `0.15` | Minimum margin between 1st and 2nd place |
