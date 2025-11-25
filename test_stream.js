#!/usr/bin/env node
/**
 * Test streaming chat completions API with multi-turn conversation support
 */

const readline = require('readline');
const { randomUUID } = require('crypto');

class ChatSession {
    constructor(port = 24000, sessionId = null) {
        this.port = port;
        this.url = `http://localhost:${port}/v1/chat/completions`;
        this.sessionId = sessionId || `test-${randomUUID()}`;
        this.messages = [];

        console.log('='.repeat(60));
        console.log('Chat Session Started');
        console.log('='.repeat(60));
        console.log(`Session ID: ${this.sessionId.substring(0, 30)}...`);
        console.log(`Port: ${port}`);
        console.log('='.repeat(60));
        console.log('\nCommands:');
        console.log("  - 'quit', 'exit', 'bye': Exit");
        console.log("  - 'clear', 'reset': Clear chat history");
        console.log("  - 'session': Show session info");
        console.log("  - 'history': Show message history");
        console.log();
    }

    async sendMessage(userMessage) {
        // Add session ID in system message
        const messages = [
            { role: 'system', content: `session_id: ${this.sessionId}` },
            ...this.messages,
            { role: 'user', content: userMessage }
        ];

        console.log(`User: ${userMessage}`);
        process.stdout.write('Agent: ');

        let fullResponse = '';

        try {
            const response = await fetch(this.url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: messages,
                    stream: true,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    if (line.includes('[DONE]')) continue;

                    try {
                        const data = JSON.parse(line.slice(6));
                        const content = data.choices?.[0]?.delta?.content;
                        if (content) {
                            process.stdout.write(content);
                            fullResponse += content;
                        }
                    } catch (e) {
                        // Skip invalid JSON
                    }
                }
            }

            console.log('\n');

            // Update message history
            this.messages.push({ role: 'user', content: userMessage });
            this.messages.push({ role: 'assistant', content: fullResponse });

            return fullResponse;

        } catch (error) {
            console.error(`\nError: ${error.message}`);
            return '';
        }
    }

    async clearHistory() {
        this.messages = [];

        // Reset session on server
        try {
            const resetUrl = `http://localhost:${this.port}/v1/sessions/${this.sessionId}/reset`;
            await fetch(resetUrl, { method: 'POST' });
            console.log('✓ Chat history cleared\n');
        } catch (error) {
            console.log(`Warning: Could not reset session on server: ${error.message}\n`);
        }
    }

    showSessionInfo() {
        console.log('\n' + '='.repeat(60));
        console.log('Session Information');
        console.log('='.repeat(60));
        console.log(`Session ID: ${this.sessionId}`);
        console.log(`Message Count: ${this.messages.length}`);
        console.log(`API Endpoint: ${this.url}`);
        console.log('='.repeat(60) + '\n');
    }

    showHistory() {
        if (this.messages.length === 0) {
            console.log('\nNo message history\n');
            return;
        }

        console.log('\n' + '='.repeat(60));
        console.log('Message History');
        console.log('='.repeat(60));
        this.messages.forEach((msg, i) => {
            const role = msg.role.toUpperCase();
            const content = msg.content.length > 100
                ? msg.content.substring(0, 100) + '...'
                : msg.content;
            console.log(`${i + 1}. [${role}] ${content}`);
        });
        console.log('='.repeat(60) + '\n');
    }

    async chatLoop() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        const question = (prompt) => new Promise((resolve) => {
            rl.question(prompt, resolve);
        });

        try {
            while (true) {
                const userInput = (await question('You: ')).trim();

                if (!userInput) continue;

                // Handle commands
                const lowerInput = userInput.toLowerCase();

                if (['quit', 'exit', 'bye'].includes(lowerInput)) {
                    console.log('Goodbye!');
                    break;
                }

                if (['clear', 'reset'].includes(lowerInput)) {
                    await this.clearHistory();
                    continue;
                }

                if (lowerInput === 'session') {
                    this.showSessionInfo();
                    continue;
                }

                if (lowerInput === 'history') {
                    this.showHistory();
                    continue;
                }

                // Send message
                await this.sendMessage(userInput);
            }
        } finally {
            rl.close();
        }
    }
}

async function main() {
    const args = process.argv.slice(2);
    let port = 24000;

    // Check for help
    if (args[0] === '--help') {
        console.log('Usage: node test_stream.js [port] [message]');
        console.log('\nOptions:');
        console.log('  port      API server port (default: 24000)');
        console.log('  message   Single message to send (interactive mode if not provided)');
        console.log('\nExamples:');
        console.log('  node test_stream.js                    # Interactive mode');
        console.log('  node test_stream.js 24000              # Interactive mode on port 24000');
        console.log('  node test_stream.js 24000 "안녕하세요"  # Single message');
        process.exit(0);
    }

    // Parse arguments
    if (args.length > 0) {
        const firstArg = parseInt(args[0]);
        if (!isNaN(firstArg)) {
            port = firstArg;
            if (args.length > 1) {
                // Single message mode
                const prompt = args.slice(1).join(' ');
                const session = new ChatSession(port);
                await session.sendMessage(prompt);
                return;
            }
        } else {
            // First arg is not a port, treat as message
            const prompt = args.join(' ');
            const session = new ChatSession(port);
            await session.sendMessage(prompt);
            return;
        }
    }

    // Interactive mode
    const session = new ChatSession(port);
    await session.chatLoop();
}

main().catch(console.error);
