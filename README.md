# PhishGuard AI

AI-powered phishing email detection system using a hybrid ELECTRA-CNN deep learning model with Explainable AI.

## Overview

PhishGuard AI analyzes email content and predicts whether an email is legitimate or phishing.

The system combines:

- ELECTRA transformer-based language representation
- CNN feature extraction
- Confidence-based risk classification
- URL security analysis
- Integrated Gradients Explainable AI (XAI)

## Features

### AI Email Classification
- Hybrid ELECTRA-CNN phishing detection model
- Binary classification:
  - Legitimate
  - Phishing

### Risk Assessment

Provides:
- Prediction confidence
- Risk category
- Explanation of suspicious indicators

### URL Analysis

Detects:
- Suspicious domains
- URL shorteners
- Risk keywords
- Unsafe URL patterns

### Explainable AI

Uses Integrated Gradients to show:
- Important tokens
- Factors increasing phishing probability
- Factors reducing phishing probability

## Installation

Clone repository:

```bash
git clone https://github.com/Humairafnan/PhishGuard-AI.git