import numpy as np
import random
import time

# --- 1. Definición del Entorno ---

# Mapeo de estados (casillas) a índices
states = {
    "A1": 0, "A2": 1, "A3": 2,  # Fila A
    "B1": 3, "B2": 4, "B3": 5   # Fila B
}
# Mapeo inverso para imprimir la política
state_names = {v: k for k, v in states.items()}

# Mapeo de acciones a índices
actions = {
    "ARRIBA": 0,
    "ABAJO": 1,
    "IZQUIERDA": 2,
    "DERECHA": 3
}
action_names = {v: k for k, v in actions.items()}

num_states = 6
num_actions = 4

# Estados terminales (el juego termina si llegamos aquí)
terminal_states = [states["A3"], states["B3"]]

def get_reward(state_idx):
    """Devuelve la recompensa para un estado dado."""
    if state_idx == states["A3"]:
        return 100  # Tesoro
    elif state_idx == states["B3"]:
        return -100 # Agujero
    else:
        return -1   # Costo de movimiento

def get_next_state(current_state_idx, action_idx):
    """
    Calcula el siguiente estado basado en el estado actual y la acción.
    Maneja los "muros" del laberinto (no puede salirse).
    """
    
    # Estado por defecto: quedarse en el mismo sitio si la acción es inválida (choca con muro)
    next_state_idx = current_state_idx 
    
    if action_idx == actions["ARRIBA"]:
        if current_state_idx in [3, 4, 5]: # Fila B -> Fila A
            next_state_idx = current_state_idx - 3
    elif action_idx == actions["ABAJO"]:
        if current_state_idx in [0, 1, 2]: # Fila A -> Fila B
            next_state_idx = current_state_idx + 3
    elif action_idx == actions["IZQUIERDA"]:
        if current_state_idx in [1, 2, 4, 5]: # Columnas 2 o 3 -> 1 o 2
            next_state_idx = current_state_idx - 1
    elif action_idx == actions["DERECHA"]:
        if current_state_idx in [0, 1, 3, 4]: # Columnas 1 o 2 -> 2 o 3
            next_state_idx = current_state_idx + 1
            
    return next_state_idx

# --- 2. Inicialización del Agente (Q-Learning) ---

# Inicializar la Tabla Q con ceros
# Filas = Estados, Columnas = Acciones
q_table = np.zeros((num_states, num_actions))

# Hiperparámetros
alpha = 0.1         # Tasa de Aprendizaje (Learning Rate)
gamma = 0.9         # Factor de Descuento
epsilon = 1.0       # Tasa de Exploración (Epsilon)
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay_rate = 0.999 # Tasa de decaimiento de epsilon

# Configuración del entrenamiento
num_episodes = 5000
max_steps_per_episode = 100 # Evita bucles infinitos

print("--- Iniciando Entrenamiento ---")

# --- 3. Bucle de Entrenamiento ---

for episode in range(num_episodes):
    
    # Reiniciar el entorno para un nuevo episodio
    state = states["A1"] # Siempre empezamos en A1
    done = False
    
    for step in range(max_steps_per_episode):
        
        # 3.1. Elegir una Acción (Epsilon-Greedy)
        
        # Generar un número aleatorio
        exploration_tradeoff = random.uniform(0, 1)
        
        if exploration_tradeoff < epsilon:
            # Exploración: Elegir una acción aleatoria
            action = random.randint(0, num_actions - 1)
        else:
            # Explotación: Elegir la mejor acción conocida (valor Q más alto)
            action = np.argmax(q_table[state, :])
            
        # 3.2. Tomar la Acción y Observar el Entorno
        new_state = get_next_state(state, action)
        reward = get_reward(new_state)
        
        if new_state in terminal_states:
            done = True
            
        # 3.3. Actualizar la Tabla Q (Fórmula de Bellman)
        
        # Valor Q antiguo
        q_old = q_table[state, action]
        
        # Valor Q máximo del *siguiente* estado
        # Si el nuevo estado es terminal, el valor futuro es 0
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(q_table[new_state, :])
            
        # El "objetivo" (target) al que queremos que Q(s,a) se parezca
        q_target = reward + gamma * max_future_q
        
        # Fórmula de actualización Q-Learning
        q_new = q_old + alpha * (q_target - q_old)
        
        # Asignar el nuevo valor en la tabla
        q_table[state, action] = q_new
        
        # 3.4. Moverse al siguiente estado
        state = new_state
        
        # Si el episodio terminó (llegó a A3 o B3)
        if done:
            break
            
    # Fin del episodio
    
    # Reducir epsilon para que el agente explote más y explore menos
    epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)
    
    if (episode + 1) % 500 == 0:
        print(f"Episodio {episode + 1} completado. Epsilon: {epsilon:.4f}")

print("--- Entrenamiento Finalizado ---")

# --- 4. Resultados ---

print("\n--- Tabla Q Final (Cerebro del Agente) ---")
# Usamos np.round para que sea más fácil de leer
print(np.round(q_table, 2))

print("\n--- Política Óptima (El mejor camino) ---")

# Imprimir la mejor acción para cada estado
for s_idx in range(num_states):
    
    # Si es un estado terminal, no hay política
    if s_idx in terminal_states:
        policy = "TERMINAL"
    else:
        # Encontrar la acción con el valor Q más alto para este estado
        best_action_idx = np.argmax(q_table[s_idx, :])
        policy = action_names[best_action_idx]
        
    print(f"Estado {state_names[s_idx]}: \t-> {policy}")