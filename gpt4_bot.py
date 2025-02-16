import openai

# Устанавливаем базовый URL по умолчанию (для OpenAI это обычно не меняется)
openai.api_base = "https://api.openai.com/v1"

# Вставляем ваш API ключ и идентификатор организации (для GPT-4)
openai.api_key = "sk-dfaf43362cde486d9590ff6a44e5bdc3"
openai.organization = "org-Pq3AjEI4F8tSxYmwskn3PTZw"

# Отправка запроса к модели GPT-4
response = openai.ChatCompletion.create(
    model="gpt-4",  # используем GPT-4
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Привет, как у тебя дела?"}
    ],
    max_tokens=150,
    temperature=0.7
)

# Вывод ответа модели
print(response.choices[0].message['content'])
