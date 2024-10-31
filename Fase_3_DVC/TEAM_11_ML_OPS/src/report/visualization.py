import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# columnas = df_Communication_and_Conflict_Management.columns
# n = len(columnas)

# for i in range(0, n, 5):
#     subset_columnas = columnas[i:i+5]
#     VIZ_DATA(df_Communication_and_Conflict_Management, subset_columnas)

def viz_data_bar_graph(df, columnas):

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))  # 1 fila, 5 columnas
    for ax, var in zip(axes, columnas):
        counts = df.groupby([var, 'Class']).size().unstack(fill_value=0)

        counts.plot(kind='barh', stacked=True, ax=ax, color=sns.color_palette('pastel'), edgecolor='black')

        ax.set_title(f'Distribución de {var} por Clase', fontsize=14)
        ax.set_ylabel(var)  # Variable en el eje Y
        ax.set_xlabel('Count')  # Conteo en el eje X

    plt.tight_layout()
    plt.savefig(f'reports/{var}.png')
    # plt.show()

# df_Communication_and_Conflict_Management

def viz_data_pairplot(data):
    custom_palette = {'Divorciado': 'orange', 'Casado': '#87CEEB'}
    sns.pairplot(data, hue='Class', palette=custom_palette)
    plt.show()
    
def heat_map(data):
    sns.heatmap(data,xticklabels=data.columns,
            yticklabels=data.columns)
    
def conf_matrix_plot(conf_matrix, model):    
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=model.classes_)
    disp.plot()
    #plt.show()
    plt.savefig(f'reports/matrix_conf.png')

def explore_data(data):
    print(data)
    print(data.Class.value_counts())
    print(data.info())
    print(data.describe())