# H&M Fashion Recommender System 👗

A TikTok-like recommender system for H&M fashion items using a 4-stage architecture with two-tower neural networks and LLM enhancements.

## 📋 Project Overview

This project builds a real-time personalized recommender system that:
- Recommends fashion items to H&M customers
- Uses a 4-stage pipeline (Query → Candidate Retrieval → Filtering → Ranking)
- Leverages two-tower neural networks for embedding generation
- Integrates LLMs for enhanced ranking and semantic search

## 🏗️ Project Structure

```
h&m-rec/
├── data/
│   ├── raw/                    # Raw CSV files
│   ├── processed/             # Cleaned and processed data
│   └── features/              # Feature engineered datasets
├── src/
│   ├── data/                  # Data processing modules
│   ├── models/                # ML model implementations
│   ├── features/              # Feature engineering
│   ├── inference/             # Prediction pipeline
│   └── utils/                 # Utility functions
├── notebooks/                 # Jupyter notebooks for exploration
├── config/                    # Configuration files
├── tests/                     # Unit tests
└── requirements.txt           # Python dependencies
```

## 📊 Data Files

- **articles.csv** (34MB): H&M fashion items with descriptions, categories, prices
- **customers.csv** (198MB): Customer demographics and preferences  
- **transactions_train.csv** (3.2GB): Purchase history and interactions

## 🎯 Architecture Components

1. **Two-Tower Model**: Creates embeddings for customers and items
2. **Vector Search**: Efficient similarity search for candidate retrieval
3. **Ranking Model**: CatBoost model for final item ranking
4. **LLM Integration**: Enhanced ranking and semantic search capabilities

## 🚀 Getting Started

1. **Setup Environment**: Install dependencies
2. **Data Exploration**: Understand the datasets
3. **Feature Engineering**: Create ML-ready features
4. **Model Training**: Train two-tower and ranking models
5. **Inference Pipeline**: Build real-time recommendation API
6. **LLM Enhancement**: Add semantic search and LLM ranking

## 📚 Learning Journey

This project teaches:
- ✅ Machine Learning fundamentals
- ✅ Recommender system architecture
- ✅ Neural network embeddings
- ✅ Vector databases and similarity search
- ✅ LLM integration
- ✅ Production ML pipelines

---

*Built following the architecture described in the DecodingML TikTok-like recommender tutorial* 