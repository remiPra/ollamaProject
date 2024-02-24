import ollama

stream = ollama.chat(
    # model='mistral',
    model='yarn-mistral:7b-128k',

    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)