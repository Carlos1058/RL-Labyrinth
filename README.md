# Laberinto con Q-Learning - Proyecto de Aprendizaje por Refuerzo

![Laberinto Demo](demo.gif) <!-- Si tienes un GIF de demostración -->

Un minijuego interactivo donde un agente inteligente aprende a navegar laberintos usando el algoritmo **Q-Learning**. Proyecto desarrollado para el curso de Machine Learning y Data Science.

## Características Principales
- **Generación procedural** de laberintos con obstáculos aleatorios
- **Bordes bloqueados** garantizando un entorno cerrado
- **Interfaz gráfica intuitiva** con visualización en tiempo real
- **Controles personalizables**:
  - Ajuste de tamaño del laberinto (5x5 a 20x20)
  - Control de densidad de obstáculos (10%-40%)
- **Sistema de aprendizaje**:
  - Implementación del algoritmo Q-Learning
  - Visualización del proceso de entrenamiento
  - Mostrado del camino óptimo encontrado

## Tecnologías Utilizadas
- Python 3.9+
- Bibliotecas principales:
  - `numpy` - Manipulación matricial del laberinto
  - `matplotlib` - Visualización gráfica e interfaz
  - `random` - Generación de obstáculos aleatorios


## Cómo Usar
### Configura el laberinto:
Usa los sliders para ajustar tamaño y densidad de obstáculos
Presiona "Reiniciar Juego" para generar un nuevo laberinto
Selecciona posiciones:
Haz clic en una celda libre para elegir el punto inicial (🚀)
Haz clic en otra celda libre para elegir la meta (🏆)
Entrena al agente:
Presiona "Comenzar Entrenamiento"
Observa cómo encuentra el camino óptimo (en rojo)

## Algoritmo Implementado
El sistema utiliza Q-Learning con los siguientes parámetros:
alpha (tasa de aprendizaje): 0.1
gamma (factor de descuento): 0.9
epsilon (exploración): 0.2
Episodios de entrenamiento: 500

## Recompensas:

Meta alcanzada: +100
Choque con obstáculo: -10
Paso normal: -1
