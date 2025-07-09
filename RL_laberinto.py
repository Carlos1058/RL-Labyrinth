'''
Este proyecto tiene como objetivo desarrollar un entorno de laberinto simple y aplicar un algoritmo de **Aprendizaje por Refuerzo** para enseñar a una IA a navegar desde un punto inicial hasta un objetivo.

Dada la naturaleza de este proyecto, considero que el algoritmo más adecuado para este tipo de probleas es **Q-Learning**, por su facilidad de implelentación y comprensión, su estabilidad y su relación entre la exploración y la explotación.

Por esa razón te propongo resolverlo usando ese algoritmo, aunque dejo a tu criterio si quieres resolverlo con otro algoritmo de tu elección. Siempre estaré a favor de que investigues, y expandas las habilidades propuestas por tu cuenta.

### Descripción del Laberinto:

El laberinto se representa como una matriz de dos dimensiones, donde cada elemento puede ser:
+ un camino libre (0)
+ un obstáculo (1)
+ el objetivo (G)

La tarea es desarrollar un agente que pueda aprender a encontrar el camino desde un punto de inicio hasta el objetivo evitando obstáculos.

### Creación del Laberinto

Debido a que el desafío de hoy es bastante complejo, y que el objetivo final no se trata de que sepas desarrollar laberintos, sino sistemas para resolverlos, voy a facilitar la tarea entregando en este cuaderno el código para generar nuestros laberintos.

Tu parte será la siguiente, que es diseñar y entrenar un modelo de Q-Learning para resolver el laberinto de la manera más eficiente, y luego mostrar una visualización sobre cómo lo ha hecho.

Te deseo toda la suerte del mundo, y sobre todo, que te diviertas de a montones.

'''

# Importar las librerías necesariass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
import random
import time

class LaberintoQLearning:
    def __init__(self):
        # Configuración inicial
        self.dimensiones = (10, 10)  # Tamaño por defecto
        self.porcentaje_obstaculos = 0.2  # 20% de obstáculos
        self.estado_inicial = None
        self.estado_objetivo = None
        self.Q = None
        self.obstaculos = []
        self.acciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Arriba, abajo, izquierda, derecha
        self.entrenamiento_completado = False
        self.camino = []
        
        # Configuración de la interfaz gráfica
        self.fig = plt.figure(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title('Laberinto con Q-Learning')  # Corregido aquí
        plt.subplots_adjust(bottom=0.3)
        self.ax = self.fig.add_subplot(111)
        
        # Crear controles
        self.crear_controles()
        
        # Generar laberinto inicial
        self.generar_laberinto()
        
        # Mostrar instrucciones iniciales
        self.mostrar_mensaje("Selecciona la posición inicial y luego la meta")
        
        # Conectar eventos
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.show()
    
    def crear_controles(self):
        # Botón para reiniciar
        ax_reiniciar = plt.axes([0.3, 0.11, 0.4, 0.05])
        self.btn_reiniciar = Button(ax_reiniciar, 'Reiniciar Juego')
        self.btn_reiniciar.on_clicked(self.reiniciar_juego)
        
        # Botón para comenzar entrenamiento
        ax_start = plt.axes([0.3, 0.03, 0.4, 0.05])
        self.btn_start = Button(ax_start, 'Comenzar Entrenamiento')
        self.btn_start.on_clicked(self.comenzar_entrenamiento)
        
        # Slider para tamaño del laberinto
        ax_tamano = plt.axes([0.2, 0.22, 0.6, 0.03])
        self.slider_tamano = Slider(
            ax=ax_tamano,
            label='Tamaño del Laberinto',
            valmin=5,
            valmax=20,
            valinit=10,
            valstep=1
        )
        self.slider_tamano.on_changed(self.actualizar_tamano)
        
        # Slider para densidad de obstáculos
        ax_obstaculos = plt.axes([0.2, 0.18, 0.6, 0.03])
        self.slider_obstaculos = Slider(
            ax=ax_obstaculos,
            label='Densidad de Obstáculos',
            valmin=0.1,
            valmax=0.4,
            valinit=0.2,
            valstep=0.05
        )
        self.slider_obstaculos.on_changed(self.actualizar_obstaculos)
    
    def generar_laberinto(self):
        """Genera un nuevo laberinto con obstáculos aleatorios y bordes bloqueados"""
        filas, columnas = self.dimensiones
        
        # Limpiar obstáculos anteriores
        self.obstaculos = []
        
        # Agregar bordes como obstáculos
        for i in range(filas):
            self.obstaculos.append((i, 0))  # Borde izquierdo
            self.obstaculos.append((i, columnas-1))  # Borde derecho
            
        for j in range(columnas):
            self.obstaculos.append((0, j))  # Borde superior
            self.obstaculos.append((filas-1, j))  # Borde inferior
        
        # Calcular número de obstáculos adicionales (excluyendo bordes)
        celdas_totales = filas * columnas
        celdas_borde = 2 * (filas + columnas) - 4
        celdas_interiores = celdas_totales - celdas_borde
        num_obstaculos = int(celdas_interiores * self.porcentaje_obstaculos)
        
        # Generar obstáculos aleatorios en el interior
        celdas_interiores = [(i, j) for i in range(1, filas-1) for j in range(1, columnas-1)]
        self.obstaculos.extend(random.sample(celdas_interiores, num_obstaculos))
        
        # Reiniciar estados y Q-table
        self.estado_inicial = None
        self.estado_objetivo = None
        self.Q = np.zeros((filas * columnas, len(self.acciones)))
        self.entrenamiento_completado = False
        self.camino = []
        
        # Redibujar
        self.dibujar_laberinto()
    
    def dibujar_laberinto(self):
        """Dibuja el laberinto en el gráfico"""
        self.ax.clear()
        filas, columnas = self.dimensiones
        
        # Configurar ejes
        self.ax.set_xlim(-0.5, columnas-0.5)
        self.ax.set_ylim(-0.5, filas-0.5)
        self.ax.set_xticks(np.arange(-0.5, columnas, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, filas, 1), minor=True)
        self.ax.grid(which="minor", color='black', linestyle='-', linewidth=2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Dibujar obstáculos
        for obs in self.obstaculos:
            self.ax.add_patch(patches.Rectangle(
                (obs[1]-0.5, obs[0]-0.5), 1, 1,
                facecolor='#2c3e50', edgecolor='black'
            ))
        
        # Dibujar inicio y meta si están definidos
        if self.estado_inicial:
            self.ax.text(
                self.estado_inicial[1], self.estado_inicial[0],
                'I', ha='center', va='center', fontsize=20
            )
        
        if self.estado_objetivo:
            self.ax.text(
                self.estado_objetivo[1], self.estado_objetivo[0],
                'F', ha='center', va='center', fontsize=20
            )
        
        # Dibujar camino si está definido
        for paso in self.camino:
            self.ax.add_patch(patches.Circle(
                (paso[1], paso[0]), 0.3,
                color="#34c024", alpha=0.7
            ))
        
        plt.draw()
    
    def mostrar_mensaje(self, mensaje):
        """Muestra un mensaje en la parte superior del gráfico"""
        self.ax.set_title(mensaje, fontsize=14, pad=20)
        plt.draw()
    
    def on_click(self, event):
        """Maneja los clics del usuario para seleccionar inicio y meta"""
        if event.inaxes != self.ax or self.entrenamiento_completado:
            return
        
        # Obtener posición del clic
        col, fila = int(round(event.xdata)), int(round(event.ydata))
        pos = (fila, col)
        
        # Validar posición
        if not self.es_valida(pos):
            self.mostrar_mensaje("¡Posición inválida! Elige un espacio libre.")
            return
        
        # Seleccionar inicio o meta
        if self.estado_inicial is None:
            self.estado_inicial = pos
            self.mostrar_mensaje("Ahora selecciona la posición de la meta (🏆)")
        elif self.estado_objetivo is None:
            self.estado_objetivo = pos
            self.mostrar_mensaje("¡Listo! Presiona 'Comenzar Entrenamiento'")
        else:
            self.mostrar_mensaje("Ya se seleccionaron inicio y meta")
        
        self.dibujar_laberinto()
    
    def es_valida(self, pos):
        """Verifica si una posición es válida (dentro del laberinto y no es obstáculo)"""
        return (
            0 <= pos[0] < self.dimensiones[0] and
            0 <= pos[1] < self.dimensiones[1] and
            pos not in self.obstaculos
        )
    
    def comenzar_entrenamiento(self, event):
        """Inicia el proceso de entrenamiento con Q-Learning"""
        if self.estado_inicial is None or self.estado_objetivo is None:
            self.mostrar_mensaje("¡Selecciona inicio y meta primero!")
            return
        
        self.mostrar_mensaje("Entrenando al agente...")
        plt.pause(0.1)  # Permitir que se actualice la interfaz
        
        # Parámetros de Q-Learning
        alpha = 0.1  # Tasa de aprendizaje
        gamma = 0.9  # Factor de descuento
        epsilon = 0.2  # Probabilidad de exploración
        episodios = 500  # Número de episodios de entrenamiento
        
        # Inicializar Q-table si es necesario
        if self.Q is None:
            self.Q = np.zeros((self.dimensiones[0] * self.dimensiones[1], len(self.acciones)))
        
        # Función para convertir estado a índice
        def estado_a_indice(estado):
            return estado[0] * self.dimensiones[1] + estado[1]
        
        # Entrenamiento
        for episodio in range(episodios):
            estado = self.estado_inicial
            terminado = False
            
            while not terminado:
                # Elegir acción (ε-greedy)
                if random.random() < epsilon:
                    accion_idx = random.randint(0, len(self.acciones)-1)
                else:
                    accion_idx = np.argmax(self.Q[estado_a_indice(estado)])
                
                # Aplicar acción
                accion = self.acciones[accion_idx]
                nuevo_estado = (estado[0] + accion[0], estado[1] + accion[1])
                
                # Calcular recompensa
                if not self.es_valida(nuevo_estado):
                    recompensa = -10
                    terminado = False
                    nuevo_estado = estado  # Permanece en el mismo estado
                elif nuevo_estado == self.estado_objetivo:
                    recompensa = 100
                    terminado = True
                else:
                    recompensa = -1
                    terminado = False
                
                # Actualizar Q-table
                self.Q[estado_a_indice(estado), accion_idx] += alpha * (
                    recompensa + 
                    gamma * np.max(self.Q[estado_a_indice(nuevo_estado)]) - 
                    self.Q[estado_a_indice(estado), accion_idx]
                )
                
                estado = nuevo_estado
            
            # Actualizar progreso cada 50 episodios
            if episodio % 50 == 0:
                self.mostrar_mensaje(f"Entrenando... Episodio {episodio}/{episodios}")
                plt.pause(0.01)
        
        # Entrenamiento completado
        self.entrenamiento_completado = True
        self.mostrar_mensaje("¡Entrenamiento completado! Mostrando camino...")
        
        # Mostrar el camino óptimo
        self.mostrar_camino_optimo()
    
    def mostrar_camino_optimo(self):
        """Muestra el camino óptimo aprendido por el agente"""
        if not self.entrenamiento_completado or self.estado_inicial is None or self.estado_objetivo is None:
            return
        
        # Reconstruir el camino
        estado = self.estado_inicial
        self.camino = [estado]
        
        def estado_a_indice(estado):
            return estado[0] * self.dimensiones[1] + estado[1]
        
        while estado != self.estado_objetivo:
            accion_idx = np.argmax(self.Q[estado_a_indice(estado)])
            estado = (estado[0] + self.acciones[accion_idx][0], 
                     estado[1] + self.acciones[accion_idx][1])
            self.camino.append(estado)
            
            # Actualizar visualización paso a paso
            self.dibujar_laberinto()
            plt.pause(0.1)
        
        self.mostrar_mensaje("¡Camino óptimo encontrado!")
    
    def reiniciar_juego(self, event):
        """Reinicia el juego con la configuración actual"""
        self.generar_laberinto()
        self.mostrar_mensaje("Selecciona la posición inicial y luego la meta")
    
    def actualizar_tamano(self, val):
        """Actualiza el tamaño del laberinto"""
        nuevo_tamano = int(self.slider_tamano.val)
        if nuevo_tamano != self.dimensiones[0]:
            self.dimensiones = (nuevo_tamano, nuevo_tamano)
            self.generar_laberinto()
            self.mostrar_mensaje("Selecciona la posición inicial y luego la meta")
    
    def actualizar_obstaculos(self, val):
        """Actualiza la densidad de obstáculos"""
        self.porcentaje_obstaculos = self.slider_obstaculos.val
        self.generar_laberinto()
        self.mostrar_mensaje("Selecciona la posición inicial y luego la meta")

# Iniciar la aplicación
if __name__ == "__main__":
    app = LaberintoQLearning()