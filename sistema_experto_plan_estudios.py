# ======================================
# Sistema Experto con Aprendizaje Automático - Caso 3
# Recomendación de Plan de Estudios para Ingeniería de Sistemas
# Versión interactiva para VSCode (ingreso por consola)
# ======================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ----------------------------
# 1. Generar datos sintéticos
# ----------------------------

np.random.seed(42)
n_samples = 100

# Características
promedio_matematicas = np.random.uniform(2.0, 5.0, n_samples)
tiene_trabajo = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
horas_disponibles = np.random.choice([10, 15, 20, 25, 30, 35, 40], n_samples)
prefiere_practicas = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
ha_reprobado = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

def asignar_plan(row):
    prom, trabajo, horas, practicas, reprob = row
    if trabajo == 1 and horas < 25:
        return "Flexible"
    elif reprob == 1 and horas < 30:
        return "Flexible"
    elif prom >= 4.0 and trabajo == 0 and horas >= 30:
        return "Acelerado"
    elif prom >= 3.0 and (trabajo == 0 or horas >= 25):
        return "Estándar"
    else:
        return "Híbrido"

df = pd.DataFrame({
    'promedio_matematicas': promedio_matematicas,
    'tiene_trabajo': tiene_trabajo,
    'horas_disponibles_semana': horas_disponibles,
    'prefiere_materias_practicas': prefiere_practicas,
    'ha_reprobado_previamente': ha_reprobado
})

df['plan_recomendado'] = [asignar_plan(row) for row in df.values]

# ----------------------------
# 2. Preparar datos
# ----------------------------

X = df.drop('plan_recomendado', axis=1)
y = df['plan_recomendado']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado para KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 3. Entrenar y evaluar modelos
# ----------------------------

modelos = {
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Bosque Aleatorio": RandomForestClassifier(n_estimators=100, random_state=42)
}

mejor_modelo = None
mejor_precision = 0
mejor_nombre = ""

print("Entrenando y evaluando modelos...\n")

for nombre, modelo in modelos.items():
    if nombre == "KNN":
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
    
    precision = accuracy_score(y_test, y_pred)
    print(f"{nombre}: Precisión = {precision:.2f}")
    
    if precision > mejor_precision:
        mejor_precision = precision
        mejor_modelo = modelo
        mejor_nombre = nombre

print(f"\n✅ Mejor modelo: {mejor_nombre} (Precisión: {mejor_precision:.2f})\n")

# ----------------------------
# 4. Ingreso interactivo del estudiante
# ----------------------------

print("=== Ingrese el perfil del estudiante ===")
try:
    promedio = float(input("Promedio en matemáticas (0.0 - 5.0): "))
    if not (0.0 <= promedio <= 5.0):
        raise ValueError
    trabajo = int(input("¿Tiene trabajo? (0 = No, 1 = Sí): "))
    if trabajo not in [0, 1]:
        raise ValueError
    horas = int(input("Horas disponibles por semana (ej: 20): "))
    if horas < 0:
        raise ValueError
    practicas = int(input("¿Prefiere materias prácticas? (0 = No, 1 = Sí): "))
    if practicas not in [0, 1]:
        raise ValueError
    reprobado = int(input("¿Ha reprobado materias antes? (0 = No, 1 = Sí): "))
    if reprobado not in [0, 1]:
        raise ValueError
except ValueError:
    print("\n❌ Entrada inválida. Por favor, use los formatos indicados.")
    exit()

# Crear vector de entrada
entrada = np.array([[promedio, trabajo, horas, practicas, reprobado]])

# Aplicar escalado si es KNN
if mejor_nombre == "KNN":
    entrada = scaler.transform(entrada)

# Predecir
recomendacion = mejor_modelo.predict(entrada)[0]

# ----------------------------
# 5. Mostrar resultado
# ----------------------------

print("\n" + "="*50)
print(f"🎓 Plan de estudios recomendado: **{recomendacion}**")
print("="*50)

# Descripción del plan (opcional, para contexto)
descripciones = {
    "Acelerado": "5–6 asignaturas/semestre, 8 semestres, ideal si tienes alta disponibilidad.",
    "Estándar": "4 asignaturas/semestre, 10 semestres, equilibrio entre estudio y otras actividades.",
    "Flexible": "2–3 asignaturas/semestre, hasta 14 semestres, ideal si trabajas o tienes limitaciones.",
    "Híbrido": "Carga moderada con combinación virtual/presencial, adaptable a tu ritmo."
}

print(f"\n💡 Detalle: {descripciones.get(recomendacion, 'Plan personalizado.')}")