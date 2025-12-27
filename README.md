# 🤖 DRL Trading Bot - XAUUSD

Advanced Deep Reinforcement Learning trading system for autonomous gold (XAUUSD) trading with 140+ features and multi-timeframe analysis.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

### Advanced AI Architecture
- **Algorithms**: PPO (Proximal Policy Optimization) & Dreamer (world model-based RL)
- **Framework**: Stable-Baselines3 with PyTorch backend
- **Hardware Support**: CPU, MPS (Apple Silicon), CUDA (NVIDIA GPUs)

### Comprehensive Market Intelligence (140+ Features)
- **Multi-Timeframe Analysis**: M5, M15, H1, H4, D1 integration
- **Technical Indicators**: 63 god-mode features (RSI, MACD, Bollinger, ATR, etc.)
- **Macro Market Data**: DXY, SPX, US10Y, VIX, Oil, Bitcoin, EURUSD, Silver, GLD
- **Economic Calendar**: Major events (NFP, CPI, FOMC, GDP) integration
- **Market Microstructure**: Order flow and volatility analysis
- **Sentiment Analysis**: Optional Reddit/news sentiment

### Trading Strategies
- **Standard**: Balanced risk-reward
- **Aggressive**: Higher frequency, tighter stops
- **Swing**: Position holding for larger moves

### Live Trading
- MetaTrader 5 integration
- MetaAPI cloud trading support
- Real-time execution with risk management
- Dynamic position sizing

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| Annual Return | 80-120%+ |
| Sharpe Ratio | 3.5-4.5+ |
| Max Drawdown | <8% |
| Win Rate | 60-65% |

## 🛠️ Installation

### Prerequisites
```bash
Python 3.12+
MetaTrader 5 (for live trading)
```

### Setup
```bash
# Clone repository
git clone https://github.com/zero-was-here/tradingbot.git
cd tradingbot

# Create virtual environment
python3 -m venv .
source bin/activate  # On Windows: .\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📚 Quick Start

### 1. Data Acquisition

#### Auto-fetch Macro Data
```bash
python scripts/fetch_all_data.py
```

#### Generate Economic Calendar
```bash
python scripts/generate_economic_calendar.py
```

#### Manual: Export XAUUSD from MT5
1. Open MetaTrader 5
2. Tools → History Center
3. Export XAUUSD M5/M15 data to `data/xauusd_m5.csv` and `data/xauusd_m15.csv`

### 2. Training

#### Local Training (Mac/CPU)
```bash
python train/train_ultimate_150.py --steps 1000000 --device mps --batch-size 64
```

#### GPU Training (CUDA)
```bash
python train/train_ultimate_150.py --steps 1000000 --device cuda --batch-size 128
```

#### Google Colab (Recommended for speed)
Upload `colab_train_ultimate_150.ipynb` to Google Colab Pro+ for fastest training (5-7 hours vs 6-8 days locally).

### 3. Evaluation
```bash
python evaluate_model.py --model train/ppo_xauusd_latest.zip
```

### 4. Live Trading
```bash
# MetaTrader 5
python live_trade_mt5.py

# MetaAPI (cloud)
python live_trade_metaapi.py
```

## 📁 Project Structure

```
├── train/              # Training scripts & saved models
├── features/           # Feature engineering modules
├── env/                # Custom Gymnasium trading environment
├── models/             # RL agent architectures
├── eval/               # Evaluation & backtesting
├── data/               # Market data storage
├── scripts/            # Data acquisition utilities
├── backtest/           # Backtesting engine
└── monitoring/         # Production monitoring
```

## 📖 Documentation

- [Colab Training Guide](COLAB_TRAINING_GUIDE.md) - Step-by-step cloud training instructions
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment setup
- [Free Deployment](FREE_DEPLOYMENT.md) - Deploy on free cloud services
- [Dreamer Implementation](DREAMER_IMPLEMENTATION_GUIDE.md) - World model RL technical details

## 🔬 Algorithms

### PPO (Proximal Policy Optimization)
- Stable on-policy training
- Proven performance in trading environments
- Multiple checkpoints saved during training

### Dreamer V3
- World model-based RL
- Learns environment dynamics
- Sample-efficient training
- Better long-term planning

## ⚙️ Configuration

Key parameters in training scripts:
- `--steps`: Total training steps (default: 1M)
- `--device`: cpu/mps/cuda
- `--batch-size`: Training batch size
- `--learning-rate`: PPO learning rate
- `--gamma`: Discount factor

## 🧪 Testing

```bash
# Quick environment test
python train/smoke_env.py

# Quick model test
python test_ultimate_quick.py

# Full backtest
python backtest/backtest_engine.py
```

## 📈 Results

Training progress with 2M steps completed on aggressive strategy. Models saved every 50k steps for evaluation and deployment.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading financial instruments involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Stable-Baselines3 team
- Dreamer V3 authors
- OpenAI Gymnasium
- MetaTrader 5 community

---

**Built with 🔥 by zero-was-here**
