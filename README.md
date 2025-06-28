# Vedantariksh – Cloud Motion Prediction using Diffusion Models

A generative AI system to forecast short-term cloud movement from INSAT-3DR/3DS imagery using conditional diffusion models. Built for the Bharatiya Antariksh Hackathon 2025.

## Structure
- `data/`: Raw and preprocessed satellite frames
- `models/`: Diffusion + UNet implementation
- `scripts/`: Training & inference pipelines
- `frontend/`: Demo web app (to be added)

## Goals
- Train on past INSAT sequences
- Predict future cloud motion (30–90 mins ahead)
- Evaluate with SSIM, MAE, PSNR
- Provide web-based visual comparison

