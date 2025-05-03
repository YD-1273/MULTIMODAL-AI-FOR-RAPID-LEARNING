# MULTIMODAL AI FOR RAPID LEARNING: Redefining Training for Proprietary Programming Systems

## 📌 Project Overview

This project presents a **Multimodal AI-driven chatbot** designed to assist freshers and new employees in rapidly learning **proprietary programming languages** and internal tools used within organizations.

By supporting **text**, **voice**, and **document inputs**, the system delivers a rich, interactive onboarding experience — bridging the gap left by traditional static documentation or instructor-led sessions.

## 🎯 Objectives

- Develop a chatbot that supports **text, voice, and document** based learning.
- Fine-tune a **LLM model** using a custom dataset (HTML/CSS/JS demo).
- Enable real-time code explanations, corrections, and suggestions.
- Provide an **admin dashboard** for model tuning and monitoring.
- Deploy on **Google Colab** using **Gradio** with quick-access URLs (no servers required).

## ⚙️ Features

- 💬 **Multimodal Input Support** (Text, Voice, PDF/DOCX)
- 🤖 **Fine-tuned LLM Model** (LoRA, 4-bit quantization)
- 🌐 **Gradio Web Interface** (Chat UI + Admin Panel)
- 🔧 **Quick Train Feature** (Real-time fine-tuning)
- 📊 **Chat Logs & Monitoring** (Admin view of user interactions)

## 🛠️ Technology Stack

| Component             | Technology Used                                |
|-----------------------|------------------------------------------------|
| Language Model        | meta-llama/LLaMA-2-7B-Chat + LoRA (4-bit)     |
| Dataset               | Custom HTML/CSS/JS dataset (100 categories)   |
| Libraries (Python)    | Hugging Face Transformers, PEFT, TRL, Gradio  |
| Document Processing   | pdfplumber, python-docx                       |
| Voice Processing      | OpenAI Whisper                                 |
| Deployment            | Google Colab, Gradio live URLs                |

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multimodal-ai-chatbot.git
   cd multimodal-ai-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Login to Hugging Face CLI (for model download):
   ```bash
   huggingface-cli login
   ```
   
## 💡 Usage

### Launch Chatbot
  ```bash
  python app.py
  ```

This starts the Gradio interface. You can:
- Type questions about code
- Upload PDF/DOCX files with code queries
- Speak your question via microphone

### Admin Dashboard Features
- Adjust model parameters (temperature, max tokens)
- Trigger Quick Train (quick_train.py) for fast re-tuning
- View recent chat logs for monitoring

### 📊 Results
- Achieved >90% accuracy on HTML/CSS queries (test dataset)
- Real-time response (<2s latency on Google Colab T4 GPU)
- Significantly reduced onboarding time for proprietary systems (proof-of-concept)

### 📈 Future Enhancements
- Expand dataset to include Python, Java, proprietary languages
- Add persistent user database for tracking progress
- Integrate image/OCR support (for screenshots)
- Enable video-based recommendations using CLIP or ViT
- Deploy as containerized enterprise solution (Docker)

### 👥 Authors
|Name	                    |ID	        |Role                         |
|-------------------------|-----------|-----------------------------|
|Yogesh Kumar Dharmsalani	|21BCE0402	|Model Training & UI          |
|Surya Venkatesh	        |21BCE0855	|Voice & Document Integration |
|Biswayan Mandal	        |21BDS0024	|Testing & Deployment         |
|Dr. Anuradha J           |Supervisor	|VIT SCOPE                    |

### 📄 License
This project is developed as part of the **B.Tech Capstone Project at VIT Vellore** and is intended for academic and demonstration purposes.
