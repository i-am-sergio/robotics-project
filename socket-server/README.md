# Servicio: socket-server

## 1. Información General

### Tecnologías Utilizadas

El servicio está construido sobre el entorno de ejecución **Node.js**, aprovechando su arquitectura orientada a eventos y su capacidad para manejar concurrencia mediante I/O no bloqueante, lo cual es ideal para aplicaciones de tiempo real.

### Dependencias

El proyecto utiliza las siguientes dependencias principales:

- **ws:** Una biblioteca de cliente y servidor WebSocket para Node.js, utilizada para establecer canales de comunicación bidireccional de baja latencia.
- **Módulos nativos de Node.js:**
- `http`: Para la creación del servidor base.
- `url` y `path`: Para la manipulación de rutas de archivos y directorios en el entorno de módulos ES6 (ECMAScript Modules).

### Propósito General

Este módulo actúa como un servidor intermediario (middleware) de comunicación en tiempo real. Su función principal es servir de puente entre un cliente de inteligencia artificial (el agente DQN implementado en C++/CUDA) y dar comandos al robot. Permite el intercambio instantáneo de comandos y estados del sistema.

### Objetivo Logrado

El objetivo principal es centralizar la gestión de conexiones y la distribución de mensajes. El servidor logra:

1. Unificar la comunicación entre distintos clientes (sensores, agentes de IA, interfaces de usuario) mediante el protocolo WebSocket.
2. Validar y procesar comandos de control de movimiento ('Arriba', 'Abajo', 'Izquierda', 'Derecha').
3. Sincronizar el estado de todos los clientes conectados mediante un mecanismo de _broadcasting_ (difusión), asegurando que el agente de C++/CUDA y la interfaz gráfica operen bajo las mismas instrucciones.

---

## 2. Análisis Técnico

### Configuración e Inicialización del Servidor

```javascript
const server = createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end("WebSocket Server for C++/CUDA DQN\n");
});

const wss = new WebSocketServer({ server });
```

Se instancia un servidor HTTP básico que responde con un mensaje de estado plano para verificaciones de salud (health checks). Posteriormente, se inicializa el servidor WebSocket (`wss`) utilizando la instancia HTTP existente. Esto permite que ambos protocolos (HTTP y WS) compartan el mismo puerto de red, facilitando el despliegue y la configuración de red.

### Gestión de Conexiones Entrantes

```javascript
wss.on("connection", (ws, req) => {
  const clientIp = req.socket.remoteAddress;
  const clientPort = req.socket.remotePort;
  const clientId = `${clientIp}:${clientPort}`;

  connections.add(ws);

  ws.send(
    JSON.stringify({
      type: "welcome",
      message: "Connected to WebSocket Server",
      clientId: clientId,
      timestamp: new Date().toISOString(),
    })
  );
  // ... (lógica de mensajes)
});
```

Este bloque define el manejador de eventos para nuevas conexiones. Cuando un cliente se conecta, se identifica mediante su dirección IP y puerto, generando un identificador único (`clientId`). La conexión se almacena en una estructura de datos tipo `Set` para mantener un registro de clientes activos y se envía un mensaje de bienvenida en formato JSON para confirmar el establecimiento exitoso de la sesión.

### Procesamiento y Validación de Mensajes

```javascript
ws.on("message", (data) => {
  try {
    let message;
    try {
      message = JSON.parse(data.toString());
    } catch {
      message = { command: data.toString() };
    }

    const validCommands = ["Arriba", "Abajo", "Izquierda", "Derecha"];

    if (message.command && validCommands.includes(message.command)) {
      processCommand(message.command, clientId, message);

      broadcast(
        JSON.stringify({
          type: "action_command",
          command: message.command,
          source: clientId,
          timestamp: new Date().toISOString(),
        })
      );

      ws.send(
        JSON.stringify({
          type: "command_ack",
          command: message.command,
          status: "broadcasted",
          timestamp: new Date().toISOString(),
          clientId: clientId,
        })
      );
    }
    // ... (manejo de heartbeats y errores)
  } catch (error) {
    // ... (manejo de excepciones)
  }
});
```

Esta sección contiene la lógica central de recepción de datos. El servidor intenta interpretar los datos entrantes como JSON. Si la estructura es válida, verifica si el comando recibido pertenece a la lista blanca de instrucciones permitidas (`validCommands`).
Si el comando es legítimo:

1. Se invoca la función de procesamiento interno.
2. Se retransmite (broadcast) el comando a todos los demás clientes conectados para mantener la sincronización.
3. Se envía una confirmación (ACK) específica al remitente indicando que la instrucción fue difundida.

### Función de Procesamiento Lógico

```javascript
function processCommand(command, clientId, metadata = {}) {
  switch (command) {
    case "Arriba":
      // Lógica para mover arriba
      break;
    case "Abajo":
      // Lógica para mover abajo
      break;
    // ... otros casos
  }
}
```

Esta función actúa como un controlador para la lógica de negocio específica. Aunque actualmente imprime logs en la consola, está diseñada como el punto de integración donde se conectarían las llamadas a las funciones nativas o la lógica de control del agente DQN en C++/CUDA. Permite segregar la lógica de comunicación (WebSocket) de la lógica de la aplicación (Movimiento/IA).

### Mecanismo de Difusión (Broadcast)

```javascript
function broadcast(message) {
  connections.forEach((client) => {
    if (client.readyState === client.OPEN) {
      client.send(message);
    }
  });
}
```

Implementa el patrón de mensajería "publish-subscribe" de manera simplificada. Itera sobre la colección de conexiones activas almacenadas en memoria y envía el mensaje proporcionado a cada cliente cuyo estado de conexión sea `OPEN`. Esto es fundamental para asegurar que todos los componentes del sistema distribuido reciban las actualizaciones de estado en tiempo real.

### Cierre del Servidor (Graceful Shutdown)

```javascript
process.on("SIGINT", () => {
  wss.clients.forEach((client) => {
    client.close();
  });
  server.close(() => {
    process.exit(0);
  });
});
```

Maneja la señal de interrupción del sistema (comúnmente CTRL+C). Asegura que, antes de terminar el proceso de Node.js, se cierren explícitamente todas las conexiones WebSocket activas y se detenga el servidor HTTP, liberando los puertos y recursos del sistema operativo de manera ordenada.

---

## 3. Salida de Ejecución

```text
✅ Server listening on http://localhost:5555
✅ WebSocket available on ws://localhost:5555
📋 Waiting for C++/CUDA client connections...
--- New connection: ::1:56789
📥 Received from ::1:56789: { command: 'Arriba' }
Valid command from ::1:56789: Arriba
- Processing command: Arriba from ::1:56789
⬆️  Mover hacia ARRIBA
📥 Received from ::1:56789: { command: 'Izquierda' }
Valid command from ::1:56789: Izquierda
- Processing command: Izquierda from ::1:56789
⬅️  Mover hacia IZQUIERDA
```

---

## 4. Instalación y Ejecución

Para poner en marcha el servidor WebSocket, siga los pasos a continuación:

### Instalar Dependencias

Instale la librería `ws` requerida para el protocolo WebSocket ejecutando:

```sh
npm install ws
```

### Ejecutar el Servidor

Inicie el servidor mediante el siguiente comando:

```sh
node server.js
```
