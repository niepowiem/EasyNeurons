# 🧠 EasyNeurons: Neural Networks from Scratch

[English Version](#english-version) | [Polska Wersja](#polska-wersja)

---

<a name="english-version"></a>
## 🇺🇸 English Version

**EasyNeurons** is an educational Python library built to demonstrate how artificial intelligence works at its most fundamental level. Forget complex frameworks with hidden logic—here, we build everything from zero using only **NumPy**.

### 🚀 Key Features
* **Built from Scratch:** No PyTorch, no TensorFlow. Just pure Python and NumPy.
* **Educational First:** The code is written to be read and understood, not just executed.
* **Modular Architecture:** Create layers, manage weights, and understand the flow of data through a network.
* **Mathematical Transparency:** Every dot product and bias addition is visible and explained.

### 📺 Watch the Tutorials
I am building this library live on my YouTube channel. If you want to understand the "why" behind the code, join me there!

👉 **My YouTube Channel** TBA

### 🛠 Quick Start
```python
import numpy as np
from easyneurons import NLayer

# Initialize a layer with 3 inputs and 5 neurons
layer1 = NLayer(n_inputs=3, n_neurons=5)

# Forward pass with some sample data
inputs = np.array([[1, 2, 3]])
layer1.forward(inputs)

print(layer1.output)
```

<a name="polska-wersja"></a>
## 🇵🇱 Polska Wersja

**EasyNeurons** to edukacyjna biblioteka Pythona stworzona, aby pokazać, jak sztuczna inteligencja działa na swoim najbardziej podstawowym poziomie. Zapomnij o skomplikowanych frameworkach z ukrytą logiką – tutaj budujemy wszystko od zera, używając wyłącznie **NumPy**.

### 🚀 Kluczowe Funkcje
* **Budowane od Podstaw:** Brak PyTorcha, brak TensorFlow. Tylko czysty Python i NumPy.
* **Priorytet Edukacyjny:** Kod jest napisany tak, aby był czytelny i zrozumiały, a nie tylko wykonywalny.
* **Modularna Architektura:** Twórz warstwy, zarządzaj wagami i zrozum przepływ danych przez sieć.
* **Przejrzystość Matematyczna:** Każdy iloczyn skalarny i dodanie biasu jest widoczne i wyjaśnione.

### 📺 Oglądaj Tutoriale
Buduję tę bibliotekę na żywo na moim kanale YouTube. Jeśli chcesz zrozumieć „dlaczego” za kodem kryje się taka, a nie inna logika, zapraszam!

👉 **Mój kanał na Youtube (Polski):** https://www.youtube.com/watch?v=W-tN-7qrv0k&list=PLrqG80ltdj3tyIQyItlQT2qBqRyy0yUok

### 🛠 Szybki Start
```python
import numpy as np
from easyneurons import NLayer

# Inicjalizacja warstwy: 3 wejścia i 5 neuronów
layer1 = NLayer(n_inputs=3, n_neurons=5)

# Przekazanie przykładowych danych (forward pass)
inputs = np.array([[1, 2, 3]])
layer1.forward(inputs)

print(layer1.output)
