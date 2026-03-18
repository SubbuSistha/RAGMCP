## UV Setup

### Install UV Globally 

#### On Powershell
- irm https://astral.sh/uv/install.ps1 | iex
- verify version uv --version

## UV Commands
- uv sync : creates .env and install dependencies

## Run following command to chunk, embedd and use chatbot
- uv run python src/basic/chunk.py
- uv run python src/basic/embedding.py
- uv run streamlit run consumer/chatbot.py
