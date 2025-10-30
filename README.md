📦 Kraljic Matrix Procurement Classifier

This Streamlit application implements a machine learning model to classify procurement items into one of the four Kraljic Matrix categories: Strategic, Leverage, Bottleneck, or Non-Critical. The classification is based on quantitative (e.g., Cost, Volume, Lead Time) and qualitative (e.g., Risk Scores) item characteristics.

The core model, a Gaussian Naive Bayes classifier, was trained on the provided realistic_kraljic_dataset.csv data.

🚀 Features

Item Classification: Instantly predicts the optimal Kraljic category for a new item.

Intuitive Input: Uses Streamlit sliders and input boxes to capture procurement characteristics.

Strategic Guidance: Provides immediate, actionable sourcing strategy recommendations based on the predicted category.

Model Transparency: Displays the model type (Gaussian Naive Bayes) and training accuracy.

⚙️ Installation and Setup

Follow these steps to get the application running on your local machine.

Prerequisites

You need Python installed (version 3.8+ is recommended).

Steps

Clone the Repository:

git clone <your-repository-url>
cd <repository-name>


Create and Activate a Virtual Environment (Recommended):

# Create environment
python -m venv venv

# Activate environment (Linux/macOS)
source venv/bin/activate

# Activate environment (Windows)
.\venv\Scripts\activate


Install Dependencies:
Install all required packages using the provided requirements.txt file.

pip install -r requirements.txt


Add Data File:
Ensure the training data file, realistic_kraljic_dataset.csv, is placed in the root directory of the project, next to kraljic_predictor_app.py.

▶️ How to Run the App

Once the dependencies are installed and the data file is in place, run the Streamlit application from your terminal:

streamlit run kraljic_predictor_app.py


The application will automatically open in your default web browser (usually at http://localhost:8501).

📊 The Kraljic Matrix

The classification provides a clear framework for procurement strategy:

Category

Profit Impact (Cost/Revenue)

Supply Risk (Complexity/Availability)

Sourcing Strategy Focus

Strategic (💎)

High

High

Partnership, long-term security, joint R&D.

Leverage (📈)

High

Low

Competitive bidding, maximize volume, price negotiation.

Bottleneck (⚠️)

Low

High

Secure supply, contingency planning, volume management.

Non-Critical (⚙️)

Low

Low

Process efficiency, automation, catalogue ordering.

🛠️ Project Structure

kraljic_predictor_app.py: The main Streamlit application script containing data loading, model training (using your real data), and the UI.

realistic_kraljic_dataset.csv: The dataset used for training the classifier.

requirements.txt: List of dependencies.

Created by Gemini
