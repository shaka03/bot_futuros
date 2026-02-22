#%% librerías
import joblib
import pandas as pd
import os
import spacy

nlp = spacy.load("es_core_news_sm")

#%% Configuración
class Config:
    MODEL_PATH = os.path.join(os.getcwd(), "src/models/news_model.joblib")
    VECTORIZER_PATH = os.path.join(os.getcwd(), "src/models/vectorizer.joblib")
    KEEP_WORDS = {
        "no", "ni", "nunca", "jamás", "jamas", "tampoco", "nada", "nadie",
        "ningún", "ninguna", "ninguno", "sin",
        "sí", "si", "también", "tambien", "siempre", "claro",
        "pero", "aunque", "más", "mas", "menos", "muy", "poco", "mucho"
    }

#%% Funciones

def limpiar_stopwords(texto):
    if not isinstance(texto, str):
        return texto
    doc = nlp(texto)
    tokens = [
        token.text for token in doc
        if not token.is_stop or token.text.lower() in Config.KEEP_WORDS
    ]
    return " ".join(tokens)

def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo no se encontró en la ruta: {model_path}")
    model = joblib.load(model_path)
    return model

def load_vectorizer(vectorizer_path: str):
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"El vectorizador no se encontró en la ruta: {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer    

# Función principal para procesar noticias
def process_news(df: pd.DataFrame) -> None:
    # Cargar el modelo entrenado
    model = load_model(Config.MODEL_PATH)
    vectorizer = load_vectorizer(Config.VECTORIZER_PATH)

    # Limpiar stopwords en los títulos de las noticias
    df["Titulo_norm"] = df["Titulo_norm"].apply(limpiar_stopwords)

    # Limpiar duplicados
    df = df.drop_duplicates(subset=["Titulo_norm"], keep="first")

    # Transformar los títulos de las noticias usando el vectorizador
    X = vectorizer.transform(df["Titulo_norm"])
    proba = model.predict_proba(X)[:,1]

    # Agregar las probabilidades al DataFrame original
    df["Tipo_noticia"] = proba

    return df
