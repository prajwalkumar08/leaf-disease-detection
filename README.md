# 🌿 Plant Leaf Disease Detection

This project is a deep learning-based web application that identifies plant leaf diseases using three powerful neural network models: **LeNet (CNN)**, **Inception**, and **DenseNet121**. It is deployed online using [Hugging Face Spaces](https://huggingface.co/spaces/Prajwalkumar0804/Leaf_Disease_Detection) with a clean Gradio UI.

## 🔗 Live Demo

👉 [Click to Try the App](https://huggingface.co/spaces/Prajwalkumar0804/Leaf_Disease_Detection)

---

## 🧠 Models Used

- ✅ **CNN (LeNet)** - Lightweight convolutional neural network
- ✅ **Inception** - Deeper model with better hierarchical features
- ✅ **DenseNet121** - Fine-tuned for high-accuracy classification

All three models are trained to classify a wide variety of plant leaf diseases across:
- 🍌 Banana
- 🥭 Mango
- 🥥 Coconut
- 🍈 Jackfruit
- 🌰 Cashew
- 🌿 Cassava

---

## 📦 Features

- Upload any leaf image (`.jpg`, `.png`, etc.)
- Get predictions from all 3 models
- View predicted class and confidence score
- Clean, responsive Gradio interface

---

## 🛠️ Technologies

| Tool        | Use                             |
|-------------|----------------------------------|
| TensorFlow  | Model loading and prediction     |
| Gradio      | Web UI for interactive inference |
| NumPy       | Image preprocessing              |
| Hugging Face | Hosting + deployment            |

---

## 🧑‍💻 Run Locally (Optional)

> Clone and run the app on your machine

```bash
git clone https://github.com/YOUR_USERNAME/Leaf_Disease_Detection.git
cd Leaf_Disease_Detection

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
