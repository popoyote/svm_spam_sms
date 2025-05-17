import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(df):
    print("Колонки в датасете:", df.columns.tolist())
    print("\nПервые строки:")
    print(df.head())

    print("\nРаспределение классов (Category):")
    print(df['Category'].value_counts())

    plt.figure(figsize=(6, 4))
    sns.countplot(x='Category', data=df)
    plt.title('Распределение спам/не спам')
    plt.xticks([0, 1], ['Не спам (ham)', 'Спам (spam)'])  # если категории закодированы как 0 и 1
    plt.show()

    print("\nДлина сообщений:")
    df['message_length'] = df['text'].apply(len)
    print(df['message_length'].describe())  # Выведем статистику, чтобы проверить данные

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True)
    plt.title('Длина сообщений по классам')
    plt.xlabel('Длина сообщения')
    plt.ylabel('Количество')
    plt.legend(['Не спам', 'Спам'])
    plt.tight_layout()  # Чтобы всё помещалось
    plt.show()  # ОБЯЗАТЕЛЬНО!


if __name__ == "__main__":
    df = pd.read_csv('data/spam.csv', encoding='latin-1', dtype=str)
    df = df[['Category', 'Message']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    perform_eda(df)
