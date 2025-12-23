from llm_clients import client_from_env

llm = client_from_env(model="gemma3:1b",provider="ollama")
response = llm.generate("Hello?")
print(response)