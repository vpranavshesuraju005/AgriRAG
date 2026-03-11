# AgriRAG: Precision Farming AI

![Python Badge](https://img.shields.io/badge/Python-3.8%2B-blue)
![Django Badge](https://img.shields.io/badge/Django-Web%20Framework-green)
![AI Badge](https://img.shields.io/badge/AI-Groq%20%7C%20LLaMA3-purple)
![Status Badge](https://img.shields.io/badge/Status-Active-brightgreen)

**AgriRAG** is a premium precision farming dashboard that leverages Machine Learning (Random Forest & LSTM) and Retrieval-Augmented Generation (RAG) to provide farmers with data-driven crop recommendations, yield estimations, and context-aware advisory.

## 🌟 Key Features

*   **Intelligent Crop Recommendation:** Uses a Random Forest classifier trained on varied soil types (Black, Clay, Loamy, Sandy) and climatic seasons to recommend the ideal crop based on NPK metrics.
*   **Yield & Price Prediction:** Employs an LSTM neural network model to estimate crop yield and predict realistic market prices.
*   **AgriRAG Assistant (Chatbot):** A highly contextual, multilingual Groq-powered AI. It acts as an agricultural consultant, recalling your latest crop predictions and prioritizing concrete data from real-world pricing and crop datasets over generic advice.
*   **Data Management Hub:** Easily upload, process, and analyze custom agricultural datasets (CSV structure), complete with automated normalization and label encoding.
*   **Sleek Glassmorphism UI:** Modern, responsive, and visually stunning web interface designed for optimal user experience.

## 🛠️ Technology Stack

*   **Backend:** Python, Django
*   **Machine Learning:** Scikit-Learn, TensorFlow/Keras, Pandas, NumPy
*   **AI Engine:** Groq API (LLaMA-3.3-70b-versatile)
*   **Frontend:** HTML5, CSS3 

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   MySQL (Optional, for production-grade user management)
*   [Groq API Key](https://console.groq.com/keys) (Required for the AgriRAG Chatbot)

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/vpranavshesuraju005/AgriRAG.git
    cd AgriRAG
    ```

2.  **Environment Setup**
    Create a `.env` file in the root configuration folder and add your Groq API key:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

3.  **Run the automated script (Windows)**
    The project includes an intelligent `run.bat` script that handles virtual environments, dependency installation, and server startup.
    ```cmd
    run.bat
    ```

4.  **Manual Installation (Alternative)**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python manage.py makemigrations
    python manage.py migrate
    python manage.py runserver 8080
    ```

5.  **Access the Dashboard**
    Open your browser and navigate to: `http://127.0.0.1:8080/`

## 📁 Dataset Requirements

If utilizing the Custom Data Upload feature, ensure your `.csv` file contains the following columns for accurate parsing:
`N`, `P`, `K`, `soil`, `season`, `label`

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/vpranavshesuraju005/AgriRAG/issues).

---
*Built to empower the future of farming.*
