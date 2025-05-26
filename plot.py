import pandas as pd
import matplotlib.pyplot as plt


def plot_histogram(file_path, title, merge_classes=False):
    df = pd.read_csv(file_path)

    if merge_classes:
        no_dr = (df['level'] == 0).sum()
        dr = (df['level'].isin([1, 2, 3, 4])).sum()
        labels = ['NO DR', 'DR']
        values = [no_dr, dr]

        print(f"\n{title}")
        print(f"NO DR (klasa 0): {no_dr}")
        print(f"DR (klasy 1–4): {dr}")

        plt.figure(figsize=(5, 4))
        bars = plt.bar(labels, values)
        plt.title(title)
        plt.ylabel('Liczba przykładów')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 100, str(yval),
                     ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.show()

    else:
        class_counts = df['level'].value_counts().sort_index()
        class_labels = {
            0: "klasa 0: NO DR",
            1: "klasa 1: MILD",
            2: "klasa 2: MODERATE",
            3: "klasa 3: SEVERE",
            4: "klasa 4: PROLIFERATIVE"
        }

        print(f"\n{title}")
        for cls, count in class_counts.items():
            print(f"{class_labels[cls]} → {count} przykładów")

        plt.figure(figsize=(8, 4))
        bars = plt.bar([class_labels[i] for i in class_counts.index], class_counts.values)
        plt.title(title)
        plt.ylabel('Liczba przykładów')
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)


        plt.tight_layout()
        plt.show()


plot_histogram('data/trainLabels.csv', 'Histogram klas - train.csv')
plot_histogram('data/train_balanced.csv', 'NO DR vs DR - train_balanced.csv', merge_classes=True)
