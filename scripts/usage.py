from dotenv import load_dotenv
load_dotenv(override=True)
import os
from llm_clients import client_from_env

llm = client_from_env(model=os.getenv("model"),provider="ollama")
response = llm.generate("Hello?")
print(response)