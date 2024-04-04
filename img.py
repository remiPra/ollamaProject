import ollama



with open('exemple.jpg', 'rb') as file:
  response = ollama.chat(
    model='llava',
    messages=[
      {
        'role': 'user',
        'content': 'is it barack obama?',
        'images': [file.read()],
      },
    ],
  )
print(response['message']['content'])