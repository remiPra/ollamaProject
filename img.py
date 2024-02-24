import ollama



with open('remiface.png', 'rb') as file:
  response = ollama.chat(
    model='llava',
    messages=[
      {
        'role': 'user',
        'content': 'describe image ',
        'images': [file.read()],
      },
    ],
  )
print(response['message']['content'])