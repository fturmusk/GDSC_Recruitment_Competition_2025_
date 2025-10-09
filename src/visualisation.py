import matplotlib.pyplot as plt
import seaborn as sns

def train_histplot(numerics_columns,train_transform):
    for col in numerics_columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(train_transform[col], kde = True, color='skyblue', edgecolor = "black")
        plt.title(f"Distribution of {col}", fontsize = 12)
        plt.xlabel(col, fontsize = 10)
        plt.ylabel("Count",fontsize = 10)
        plt.grid(axis = "y", linestyle = "--", alpha = 0.5)
        plt.tight_layout()
        plt.show()

        print(train_transform[col].describe())

def train_subplot(numerics_columns, train_transform):
    plt.figure(figsize=(20,30))
    for i, col in enumerate(numerics_columns):
        plt.subplot(9,5,i+1)
        sns.boxplot(data = train_transform, y = col, color = '#FFA728')
        plt.title(f"Boxplot: {col}")
        plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_contplot(object_columns, train_transform, df):
    for col in object_columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data = train_transform, x = col, order = train_transform[col].value_counts().index, palette = "Set2", edgecolor = "black")
        plt.title(f"Distribution of {col}", fontsize = 12)
        plt.xlabel(col, fontsize = 10)
        plt.ylabel("Count",fontsize = 10)
        plt.grid(axis = "y", linestyle = "--", alpha = 0.5)
        plt.tight_layout()
        plt.show()
        print(df[col].value_counts(normalize=True)*100)