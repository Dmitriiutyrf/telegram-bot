import openai

# Настраиваем базовый URL для DeepSeek
openai.api_base = "https://api.deepseek.com/v1"
# Вставляем ваш API-ключ для DeepSeek (тот же ключ, если он выдан для GPT‑4, или отдельный, если имеется)
openai.api_key = "sk-dfaf43362cde486d9590ff6a44e5bdc3"
# Устанавливаем идентификатор организации
openai.organization = "org-Pq3AjEI4F8tSxYmwskn3PTZw"

# Отправляем запрос к модели DeepSeek
response = openai.ChatCompletion.create(
    model="deepseek-chat",  # Убедитесь, что это корректное имя модели согласно документации DeepSeek
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Привет, расскажи, что такое DeepSeek?"}
    ],
    max_tokens=150,
    temperature=0.7
)

# Выводим ответ от модели
print(response.choices[0].message['content'])
