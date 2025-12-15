import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// Obtener directorio actual
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configuración del servidor
const PORT = process.env.PORT || 5555;
const HOST = process.env.HOST || 'localhost';

// Crear servidor HTTP
const server = createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('WebSocket Server for C++/CUDA DQN\n');
});

// Crear servidor WebSocket
const wss = new WebSocketServer({ server });

// Almacenar conexiones activas
const connections = new Set();

console.log(`🚀 Starting WebSocket server on ws://${HOST}:${PORT}`);

// Manejo de conexiones WebSocket
wss.on('connection', (ws, req) => {
    const clientIp = req.socket.remoteAddress;
    const clientPort = req.socket.remotePort;
    const clientId = `${clientIp}:${clientPort}`;
    
    console.log(`✅ New connection: ${clientId}`);
    connections.add(ws);
    
    // Enviar mensaje de bienvenida
    ws.send(JSON.stringify({
        type: 'welcome',
        message: 'Connected to WebSocket Server',
        clientId: clientId,
        timestamp: new Date().toISOString()
    }));
    
    // Manejar mensajes recibidos
    ws.on('message', (data) => {
        try {
            // Intentar parsear como JSON
            let message;
            try {
                message = JSON.parse(data.toString());
            } catch {
                // Si no es JSON válido, tratar como string
                message = { command: data.toString() };
            }
            
            console.log(`📥 Received from ${clientId}:`, message);
            
            // Validar que sea una instrucción válida
            const validCommands = ['Arriba', 'Abajo', 'Izquierda', 'Derecha'];
            
            if (message.command && validCommands.includes(message.command)) {
                console.log(`✅ Valid command from ${clientId}: ${message.command}`);
                
                // Procesar la instrucción (aquí iría la lógica del DQN)
                processCommand(message.command, clientId, message);
                
                // Responder con confirmación
                ws.send(JSON.stringify({
                    type: 'command_ack',
                    command: message.command,
                    status: 'processed',
                    timestamp: new Date().toISOString(),
                    clientId: clientId
                }));
            } else if (message.type === 'heartbeat') {
                // Responder a heartbeat
                ws.send(JSON.stringify({
                    type: 'heartbeat_ack',
                    timestamp: new Date().toISOString()
                }));
            } else {
                console.log(`❌ Invalid command from ${clientId}:`, message);
                ws.send(JSON.stringify({
                    type: 'error',
                    message: 'Invalid command',
                    validCommands: validCommands,
                    timestamp: new Date().toISOString()
                }));
            }
            
        } catch (error) {
            console.error(`❌ Error processing message from ${clientId}:`, error);
            ws.send(JSON.stringify({
                type: 'error',
                message: 'Error processing message',
                error: error.message,
                timestamp: new Date().toISOString()
            }));
        }
    });
    
    // Manejar cierre de conexión
    ws.on('close', () => {
        console.log(`❌ Connection closed: ${clientId}`);
        connections.delete(ws);
    });
    
    // Manejar errores
    ws.on('error', (error) => {
        console.error(`⚠️ WebSocket error for ${clientId}:`, error);
        connections.delete(ws);
    });
});

// Función para procesar comandos (aquí se integraría con el DQN)
function processCommand(command, clientId, metadata = {}) {
    console.log(`🔧 Processing command: ${command} from ${clientId}`);
    
    // Aquí se integraría con la lógica del DQN de CUDA
    // Por ahora solo mostramos la acción
    switch(command) {
        case 'Arriba':
            console.log('⬆️  Mover hacia ARRIBA');
            // Lógica para mover arriba
            break;
        case 'Abajo':
            console.log('⬇️  Mover hacia ABAJO');
            // Lógica para mover abajo
            break;
        case 'Izquierda':
            console.log('⬅️  Mover hacia IZQUIERDA');
            // Lógica para mover izquierda
            break;
        case 'Derecha':
            console.log('➡️  Mover hacia DERECHA');
            // Lógica para mover derecha
            break;
    }
    
    // Aquí podrías enviar datos al cliente CUDA/DQN
    // broadcast(JSON.stringify({
    //     type: 'action_executed',
    //     command: command,
    //     result: 'success',
    //     timestamp: new Date().toISOString()
    // }));
}

// Función para enviar mensajes a todos los clientes
function broadcast(message) {
    connections.forEach(client => {
        if (client.readyState === client.OPEN) {
            client.send(message);
        }
    });
}

// Manejar cierre limpio del servidor
process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down server...');
    wss.clients.forEach(client => {
        client.close();
    });
    server.close(() => {
        console.log('✅ Server closed');
        process.exit(0);
    });
});

// Iniciar servidor
server.listen(PORT, () => {
    console.log(`✅ Server listening on http://${HOST}:${PORT}`);
    console.log(`✅ WebSocket available on ws://${HOST}:${PORT}`);
    console.log('📋 Waiting for C++/CUDA client connections...');
});

export { wss, server, broadcast };