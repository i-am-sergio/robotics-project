#!/usr/bin/env python3
import curses
import time
import RPi.GPIO as GPIO
import asyncio
import websockets
import json
import threading
from queue import Queue
import sys

# Pines BCM
IN1 = 16  # L298N IN1
IN2 = 19  # L298N IN2
IN3 = 20  # L298N IN3
IN4 = 26  # L298N IN4

# Configuración
WS_URL = "ws://localhost:5555"
COMMAND_QUEUE = Queue()
running = True

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for p in (IN1, IN2, IN3, IN4):
        GPIO.setup(p, GPIO.OUT)
        GPIO.output(p, GPIO.LOW)

def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

def forward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

def back():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def left():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def right():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

class WebSocketRobot:
    def __init__(self):
        self.ws = None
        self.connected = False
        
    async def connect(self):
        """Conecta al servidor WebSocket"""
        try:
            print(f"Conectando a {WS_URL} ...")
            self.ws = await websockets.connect(WS_URL)
            print("-- Conectado al servidor WebSocket --")
            self.connected = True
            
            # Recibir mensaje de bienvenida
            welcome = await self.ws.recv()
            data = json.loads(welcome)
            print(f"Servidor: {data.get('message', '')}")
            
            return True
            
        except Exception as e:
            print(f"[x] Error de conexión: {e}")
            return False
    
    async def send_command(self, command):
        """Envía un comando al servidor"""
        if not self.connected or not self.ws:
            print("[x] No conectado al servidor")
            return None
            
        try:
            message = json.dumps({"command": command})
            await self.ws.send(message)
            print(f" - Enviado: {command}")
            
            # Esperar respuesta
            response = await self.ws.recv()
            data = json.loads(response)
            print(f" - Respuesta: {data.get('type', '')}")
            return data
            
        except Exception as e:
            print(f"[x] Error enviando comando: {e}")
            return None
    
    async def listen_for_broadcast(self):
        """Escucha mensajes broadcast del servidor"""
        if not self.connected or not self.ws:
            return
            
        try:
            while self.connected:
                try:
                    # Esperar mensaje con timeout
                    message = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
                    data = json.loads(message)
                    
                    # Verificar si es un comando broadcast
                    if data.get('type') == 'action_command':
                        command = data.get('command')
                        if command in ['Arriba', 'Abajo', 'Izquierda', 'Derecha']:
                            COMMAND_QUEUE.put(command)
                            print(f"📡 Comando broadcast recibido: {command}")
                    
                except asyncio.TimeoutError:
                    # Timeout normal, continuar
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("[x] Conexión cerrada por el servidor")
                    self.connected = False
                    break
                    
        except Exception as e:
            print(f"[x] Error escuchando: {e}")
    
    async def close(self):
        """Cierra la conexión"""
        if self.ws:
            await self.ws.close()
        self.connected = False

async def websocket_task():
    """Tarea asíncrona principal para WebSocket"""
    robot_ws = WebSocketRobot()
    
    while running:
        if await robot_ws.connect():
            # Iniciar escucha en segundo plano
            listen_task = asyncio.create_task(robot_ws.listen_for_broadcast())
            
            # Mantener conexión activa
            while robot_ws.connected and running:
                await asyncio.sleep(0.1)
            
            # Cancelar tarea de escucha
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass
        
        if running:
            print("!!! Reconectando en 3 segundos...")
            await asyncio.sleep(3)
    
    await robot_ws.close()

def start_websocket_thread():
    """Inicia el cliente WebSocket en un hilo separado"""
    def run_websocket():
        asyncio.run(websocket_task())
    
    thread = threading.Thread(target=run_websocket, daemon=True)
    thread.start()
    return thread

def main(stdscr):
    """Función principal con interfaz curses"""
    global running
    
    setup_gpio()
    ws_thread = start_websocket_thread()
    
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)
    
    stdscr.clear()
    stdscr.addstr(0, 0, "Robot Control - WebSocket")
    stdscr.addstr(1, 0, "Modo: Esperando comandos broadcast del servidor")
    stdscr.addstr(2, 0, "------------------------------------------------")
    stdscr.addstr(4, 0, "Teclas:")
    stdscr.addstr(5, 0, "  W/A/S/D - Control manual del robot")
    stdscr.addstr(6, 0, "  U/I/O/P - Enviar comandos al servidor")
    stdscr.addstr(7, 0, "  Q       - Salir")
    stdscr.addstr(9, 0, "Estado: ")
    stdscr.addstr(10, 0, "Último comando: ")
    stdscr.addstr(12, 0, "Log: ")
    
    log_line = 12
    last_state = "detenido"
    last_command = "ninguno"
    command_timeout = 0.3
    last_command_time = time.time()
    
    try:
        while running:
            # Procesar comandos de la cola (broadcast)
            if not COMMAND_QUEUE.empty():
                command = COMMAND_QUEUE.get()
                last_command = command
                last_command_time = time.time()
                
                if command == "Arriba":
                    forward()
                    state = "adelante"
                elif command == "Abajo":
                    back()
                    state = "atras"
                elif command == "Izquierda":
                    left()
                    state = "izquierda"
                elif command == "Derecha":
                    right()
                    state = "derecha"
                else:
                    stop()
                    state = "detenido"
                
                # Actualizar pantalla
                stdscr.addstr(10, 0, f"Último comando: {command} (broadcast)      ")
                if state != last_state:
                    stdscr.addstr(9, 0, f"Estado: {state}      ")
                    last_state = state
                    stdscr.addstr(log_line, 5, f"Broadcast: {command}          ")
                    log_line += 1
                    if log_line > 20:
                        log_line = 12
            
            # Auto-detener si ha pasado tiempo
            if time.time() - last_command_time > command_timeout:
                stop()
                if last_state != "detenido":
                    state = "detenido"
                    stdscr.addstr(9, 0, f"Estado: {state} (auto)      ")
                    last_state = state
            
            # Control manual con teclado
            key = stdscr.getch()
            
            if key in (ord('q'), ord('Q')):
                running = False
                break
            
            # Control manual del robot (solo local)
            elif key in (ord('w'), ord('W')):
                forward()
                state = "adelante (manual)"
                last_command_time = time.time()
                last_command = "manual: adelante"
            elif key in (ord('s'), ord('S')):
                back()
                state = "atras (manual)"
                last_command_time = time.time()
                last_command = "manual: atras"
            elif key in (ord('a'), ord('A')):
                left()
                state = "izquierda (manual)"
                last_command_time = time.time()
                last_command = "manual: izquierda"
            elif key in (ord('d'), ord('D')):
                right()
                state = "derecha (manual)"
                last_command_time = time.time()
                last_command = "manual: derecha"
            
            # Actualizar pantalla si hubo cambio manual
            if key in (ord('w'), ord('W'), ord('s'), ord('S'), 
                      ord('a'), ord('A'), ord('d'), ord('D')):
                if state != last_state:
                    stdscr.addstr(9, 0, f"Estado: {state}      ")
                    stdscr.addstr(10, 0, f"Último comando: {last_command}      ")
                    last_state = state
            
            stdscr.refresh()
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        running = False
    finally:
        stop()
        GPIO.cleanup()
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        print("\n-- [Robot detenido] --")

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except Exception as e:
        print(f"!!! Error: {e}")
        GPIO.cleanup()