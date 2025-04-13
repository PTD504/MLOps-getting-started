# CS317.P21---Machine Learning Operations
![UIT](https://img.shields.io/badge/from-UIT%20VNUHCM-blue?style=for-the-badge&link=https%3A%2F%2Fwww.uit.edu.vn%2F)

 <h2 align="center"> SENTIMENT ANALYSIS APPLICATION (Text-based data) </h2>

<p align="center">
  <img src="https://en.uit.edu.vn/sites/vi/files/banner_en.png" alt="Alt text">
</p>

## Contributors  
- **Phan Thanh Đăng** – [22520193@gm.uit.edu.vn](mailto:22520193@gm.uit.edu.vn)  
- **Nguyễn Thanh Hùng** – [22520518@gm.uit.edu.vn](mailto:22520518@gm.uit.edu.vn)  
- **Võ Đình Trung** – [22521571@gm.uit.edu.vn](mailto:22521571@gm.uit.edu.vn)

## Supervisors  
- **ThS. Đỗ Văn Tiến**  
  Email: [tiendv@uit.edu.vn](mailto:tiendv@uit.edu.vn)
---

## Introduction  
This project, **Sentiment Analysis**, aims to build a system that analyzes sentiment from textual data. The objective is to recognize emotions (positive, negative, and neutral) in various text inputs such as product reviews, social media comments, tweets, and customer emails. The system will compare the performance of traditional machine learning models (e.g., Logistic Regression, SVM, Naive Bayes in combination with TF-IDF) with modern deep learning approaches (e.g., LSTM, BERT, DistilBERT). Additionally, it integrates monitoring and performance tracking mechanisms to detect any model drift when deployed in a real-time application.

## Dataset  
- **[IMDb Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)**  
  The IMDb reviews dataset is used as the primary data source, providing diverse sentiments and varying text lengths.

## Models Implemented  

- **Traditional Machine Learning:**  
  - Logistic Regression  
  - SVM  
  - Naive Bayes  
  - Text processing using TF-IDF

- **Deep Learning:**  
  - LSTM  

The goal is to compare the effectiveness of traditional approaches versus transformer-based models and tune parameters such as embeddings, max features, and tokenizer settings.

## Core Features

### Essential Functionalities  
- **User Text Input:**  
  Accepts input text from users, such as product reviews, social media comments, tweets, and customer emails.

- **Sentiment Prediction:**  
  Classifies the input text into sentiment labels:
  - Positive  
  - Negative  
  - Neutral

- **Probability Display:**  
  Along with the prediction, the system shows the probability for each label (e.g., Positive: 90%, Negative: 10%), giving users an insight into the confidence of the prediction.

- **User Interface:**  
  A simple web application UI can be developed using frameworks such as Streamlit, Gradio, or a combination of React for the frontend and an API backend.

- **Model Monitoring:**  
  Tracks metrics such as accuracy and input distribution, and raises alerts when model drift is detected during real-time deployment.

### Advanced Features (Optional Enhancements)  
- **Prediction Explainability:**  
  Highlights the most influential words that affect the prediction using tools like LIME or SHAP.

- **Multilingual Support:**  
  Initially focused on English, with the possibility of expanding to other languages (e.g., Vietnamese) by employing multilingual BERT models.

- **Analysis Log History:**  
  Maintains a log of past sentiment analyses for user reference and trend analysis.

- **Time-based Sentiment Dashboard:**  
  Provides a dashboard to monitor sentiment trends over time (e.g., daily, weekly) when processing streaming data.

## Proposed Tech Stack  

| Component             | Tools/Frameworks                                                                 |
| --------------------- | -------------------------------------------------------------------------------- |
| Training Pipeline     | scikit-learn (TF-IDF + Logistic Regression) and LSTM                             |
| Experiment Tracking   | MLflow                                                                           |
| Data Versioning       | DVC                                                                              |
| CI/CD Pipeline        | GitHub Actions or Jenkins                                                        |
| Model Deployment      | Local                                                                            |
| Frontend UI           | Streamlit / Gradio / React                                                       |
| Monitoring            | Prometheus + Grafana           |

## Demo  
A demo video will illustrate how the system handles text input and produces sentiment predictions.  
*(Insert demo video link here if available)*

**Example Scenario:**
- **Input Text:** "The product was delivered extremely fast, and the packaging was excellent. Very satisfied!"  
- **Predicted Result:**  
  - Sentiment: Positive  
  - Confidence: 95%  
  - Highlight: “extremely fast”, “very satisfied”

## Installation & Setup

### Requirements  
- Python 3.8 or higher  
- Docker 

### Setup Steps  
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/PTD504/MLOps-getting-started.git
   cd MLOps-getting-started
   ```

2. **Set Up Virtual Environment and Install Dependencies:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```


### Running Instructions 
1. **Run the Training Pipeline:**
   - ```
     python pipeline.py
     ```
  - You can skip this step 1 with this local version 
2. **Run the web app locally:**
   ```bash
     python app.py
   ```

3. **Access the web interface:**
   ```bash
   Open your browser and go to http://localhost:5000
   ```
4. **View MLflow experiments:**
   ```bash
   mlflow ui
   ```
 - Then open http://localhost:5000 in your browser to view the experiment tracking dashboard.

## Project Structure  
```
MLOps-getting-started/
├── data/                   # Raw and processed datasets
├── models/                 # Trained models storage
├── mlruns/                 # Mlflow information
├── src/                    
│   ├── data/               # Data processing and DVC tracking
│   └models/                # ML and deep learning model implementations
├── templates/              # Experiment logs (managed by MLflow)
├── requirements.txt        # Required Python libraries
└── README.md
```
## Reference
```
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```
