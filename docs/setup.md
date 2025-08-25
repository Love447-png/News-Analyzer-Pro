## Development Setup
### 1. Prerequisites
- Python 3.8+ with pip
- Node.js 14+ with npm
- Git

### 2. API Keys Setup
You'll need API keys from:
- [OpenAI](https://platform.openai.com/api-keys)
- [Hugging Face](https://huggingface.co/settings/tokens)

### 3. Local Development
```bash
# Clone and setup
git clone https://github.com/yourusername/news-analyzer-pro.git
cd news-analyzer-pro

# Python setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Node.js setup
npm install

# Environment setup
cp .env.example .env
# Edit .env with your API keys

# Start development servers
npm run dev
