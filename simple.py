import ollama

stream = ollama.chat(
    # model='mistral',
    model='mistral:latest',

    messages=[{'role': 'user', 'content': 'en francais un exemple des enseignemets de buddha'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)