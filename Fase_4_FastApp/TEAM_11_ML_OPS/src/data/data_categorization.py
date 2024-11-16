# Defino una funion que a partir de los numeros me crea un df a partir de una lista
def atr_list_maker(lst):
    return ['Atr' + str(valor) for valor in lst]

def df_maker(df, list):
    atr_list = atr_list_maker(list)
    atr_list.append('Class')
    #print(atr_list)
    df_new = df[atr_list].copy()
    df_new['Class'] = df_new['Class'].replace({1: 'Divorciado', 0: 'Casado'})

    return df_new

def separation_data(data):
    Communication_and_Conflict_Management = [1, 3, 4, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    Relationship_Harmony_and_Shared_Values = [2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    Emotional_Connection_and_Bonding = [5, 8, 9, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    Dissatisfaction_and_Detachment = [6, 7, 44]
    Blame_and_Defensiveness = [48, 49, 50, 51, 52, 53, 54]
    ###
    listas_creados = {
        'Communication_and_Conflict_Management': Communication_and_Conflict_Management,
        'Relationship_Harmony_and_Shared_Values': Relationship_Harmony_and_Shared_Values,
        'Emotional_Connection_and_Bonding': Emotional_Connection_and_Bonding,
        'Dissatisfaction_and_Detachment': Dissatisfaction_and_Detachment,
        'Blame_and_Defensiveness':  Blame_and_Defensiveness
    }

    df_Communication_and_Conflict_Management = df_maker(data, Communication_and_Conflict_Management)
    df_Relationship_Harmony_and_Shared_Values = df_maker(data, Relationship_Harmony_and_Shared_Values)
    df_Emotional_Connection_and_Bonding = df_maker(data, Emotional_Connection_and_Bonding)
    df_Dissatisfaction_and_Detachment = df_maker(data, Dissatisfaction_and_Detachment)
    df_Blame_and_Defensiveness = df_maker(data, Blame_and_Defensiveness)
    ###
    dfs_creados = {
        'df_Communication_and_Conflict_Management': df_Communication_and_Conflict_Management,
        'df_Relationship_Harmony_and_Shared_Values': df_Relationship_Harmony_and_Shared_Values,
        'df_Emotional_Connection_and_Bonding': df_Emotional_Connection_and_Bonding,
        'df_Dissatisfaction_and_Detachment': df_Dissatisfaction_and_Detachment,
        'df_Blame_and_Defensiveness': df_Blame_and_Defensiveness
    }
    # Validando la creacion de los dfs
    for key_lista, lista in listas_creados.items():
        key_df = 'df_' + key_lista  # Construir la clave del DataFrame correspondiente
        df_ = dfs_creados.get(key_df)  # Obtener el DataFrame del diccionario

        if df_ is not None:
            # Imprimir el número de columnas del DataFrame y la longitud de la lista + 1
            print(f"Para df_{key_lista}:")
            print(f"Columnas en DF: {df_.shape[1]} | Valores en Lista: {len(lista)+1}" , '\n')
        else:
            print(f"No se encontró DataFrame para {key_df}.")

    return dfs_creados