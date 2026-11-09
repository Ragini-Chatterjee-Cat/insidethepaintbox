/**
 * Inside the Paintbox - AI Chat Widget
 * Connects to RAG backend to answer questions about artworks
 */

// ============================================
// CONFIGURATION - UPDATE THIS AFTER DEPLOYMENT
// ============================================
// For local development: 'http://localhost:8000'

const API_URL = 'https://art-website-production.up.railway.app';


// ============================================
// CHAT FUNCTIONALITY
// ============================================

/**
 * Generate or retrieve persistent thread ID for conversation memory
 */
function getOrCreateThreadId() {
    let threadId = localStorage.getItem('chatThreadId');
    if (!threadId) {
        // Generate a unique thread ID (UUID v4)
        threadId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
        localStorage.setItem('chatThreadId', threadId);
    }
    return threadId;
}

/**
 * Generate or retrieve a persistent visitor ID, used to remember this
 * browser's preferences across conversations. Unlike the thread ID, the
 * "new conversation" button never resets it. A browser that already has a
 * thread ID reuses it as its visitor ID, because preferences saved before
 * visitor IDs existed are stored under that thread ID.
 */
function getOrCreateVisitorId() {
    let visitorId = localStorage.getItem('chatVisitorId');
    if (!visitorId) {
        visitorId = localStorage.getItem('chatThreadId') || crypto.randomUUID();
        localStorage.setItem('chatVisitorId', visitorId);
    }
    return visitorId;
}

document.addEventListener('DOMContentLoaded', function() {
    const chatButton = document.getElementById('chatButton');
    const chatBox = document.getElementById('chatBox');
    const chatClose = document.getElementById('chatClose');
    const chatBody = document.querySelector('.chat-body');
    const chatForm = document.querySelector('.chat-form');

    // Get persistent thread ID for conversation memory
    let threadId = getOrCreateThreadId();
    const visitorId = getOrCreateVisitorId();

    // Replace the contact form with chat interface
    chatForm.innerHTML = `
        <div class="chat-input-container">
            <input
                type="text"
                id="userMessage"
                class="chat-input"
                placeholder="Want to know more about the art..."
                autocomplete="on"
            >
            <button type="button" id="sendBtn" class="chat-send-btn">
                <i class="fa fa-paper-plane"></i>
            </button>
        </div>
    `;

    // Add clear chat button to header
    const chatHeader = document.querySelector('.chat-header');
    const clearBtn = document.createElement('button');
    clearBtn.className = 'chat-clear-btn';
    clearBtn.innerHTML = '<i class="fa fa-refresh"></i>';
    clearBtn.title = 'Start new conversation';
    clearBtn.addEventListener('click', clearConversation);
    chatHeader.insertBefore(clearBtn, chatClose);

    // Add custom styles for the new chat interface
    addChatStyles();

    // Open chat
    chatButton.addEventListener('click', function() {
        chatBox.classList.add('active');
        chatButton.style.display = 'none';
        document.getElementById('userMessage').focus();
    });

    // Close chat
    chatClose.addEventListener('click', function() {
        chatBox.classList.remove('active');
        chatButton.style.display = 'flex';
    });

    // Send message on button click
    document.getElementById('sendBtn').addEventListener('click', sendMessage);

    // Send message on Enter key
    document.getElementById('userMessage').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });

    /**
     * Send message to the RAG backend with persistent memory
     */
    async function sendMessage() {
        const input = document.getElementById('userMessage');
        const message = input.value.trim();

        if (!message) return;

        // Add user message to chat
        addMessage(message, 'user');
        input.value = '';
        input.disabled = true;

        // Show typing indicator
        const typingIndicator = addTypingIndicator();

        try {
            // Call the RAG API v2 with persistent thread ID
            const response = await fetch(`${API_URL}/chat/v2`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    thread_id: threadId,
                    visitor_id: visitorId
                })
            });

            if (!response.ok) {
                let errorMsg;
                if (response.status === 400) {
                    errorMsg = "You said too much for my poor little haiku to process.";
                } else if (response.status === 429) {
                    errorMsg = "You're sending messages too quickly, chill out.";
                } else if (response.status === 500) {
                    errorMsg = "Something is wrong with me, sadness.";
                } else {
                    errorMsg = `Unexpected error (${response.status}). Please try again.`;
                }
                typingIndicator.remove();
                addMessage(errorMsg, 'bot');
                input.disabled = false;
                input.focus();
                return;
            }

            const data = await response.json();

            // Remove typing indicator and show response
            typingIndicator.remove();
            const botMessage = addMessage(data.response, 'bot');
            addArtworkThumbnails(botMessage, data.response);

        } catch (error) {
            console.error('Chat error:', error);
            typingIndicator.remove();
            addMessage(
                "Can't reach the server right now. Please check your connection and try again.",
                'bot'
            );
        }

        input.disabled = false;
        input.focus();
    }

    /**
     * Clear conversation history (start fresh)
     */
    async function clearConversation() {
        try {
            await fetch(`${API_URL}/chat/history/${threadId}`, {
                method: 'DELETE'
            });
            // Generate new thread ID for fresh conversation
            threadId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0;
                const v = c === 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
            localStorage.setItem('chatThreadId', threadId);
            // Clear chat UI
            chatBody.innerHTML = '';
            addMessage("Hello! I'm here to help you explore the artworks. What would you like to know?", 'bot');
        } catch (error) {
            console.error('Error clearing conversation:', error);
        }
    }

    /**
     * Convert URLs in text to clickable links
     */
    function linkifyText(text) {
        // Handle markdown links [text](url) first
        text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g, (_, label, url) => {
            return `<a href="${url}" target="_blank" class="chat-link-btn">${label}</a>`;
        });
        // Then handle plain URLs
        text = text.replace(/(https?:\/\/[^\s<]+)/g, (url) => {
            return `<a href="${url}" target="_blank" class="chat-link-btn">View here</a>`;
        });
        return text;
    }

    /**
     * Add a message to the chat
     */
    function addMessage(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}`;
        // For bot messages, convert URLs to clickable links
        if (type === 'bot') {
            messageDiv.innerHTML = linkifyText(text);
        } else {
            messageDiv.textContent = text;
        }
        chatBody.appendChild(messageDiv);
        chatBody.scrollTop = chatBody.scrollHeight;
        return messageDiv;
    }

    /**
     * Show a thumbnail of each artwork a bot reply links to (up to 3). The
     * reply only carries the artwork page's URL, so fetch that page from
     * this same site and use its painting (img#cub) - thumbnails then stay
     * in sync with the pages without any extra data. Netlify's image CDN
     * resizes it: a 6 MB original arrives as a ~40 KB thumbnail.
     */
    async function addArtworkThumbnails(messageDiv, text) {
        const pageUrls = text.match(/https?:\/\/[^\s<]+\/artworks\/[^\s<]+?\.html/g) || [];
        if (!pageUrls.length) return;
        const row = document.createElement('div');
        row.className = 'chat-thumbs';
        messageDiv.appendChild(row);
        for (const pageUrl of [...new Set(pageUrls)].slice(0, 3)) {
            try {
                // Fetch by path, so this also works on deploy previews and locally
                const pagePath = new URL(pageUrl).pathname;
                const html = await (await fetch(pagePath)).text();
                const page = new DOMParser().parseFromString(html, 'text/html');
                const painting = page.querySelector('img#cub') || page.querySelector('img');
                if (!painting) continue;

                const imagePath = decodeURIComponent(
                    new URL(painting.getAttribute('src'), new URL(pagePath, location.origin)).pathname
                );
                const link = document.createElement('a');
                link.href = pageUrl;
                link.target = '_blank';
                link.className = 'chat-thumb';
                const img = document.createElement('img');
                img.alt = painting.alt || '';
                img.loading = 'lazy';
                img.src = `/.netlify/images?url=${encodeURIComponent(imagePath)}&w=480`;
                // No image CDN (e.g. a local preview server): use the original
                img.onerror = () => { img.onerror = null; img.src = imagePath; };
                img.onload = () => { chatBody.scrollTop = chatBody.scrollHeight; };
                link.appendChild(img);
                row.appendChild(link);
            } catch (error) {
                // A thumbnail is decoration - never let it break the reply
                console.error('Could not load artwork thumbnail:', error);
            }
        }
    }

    /**
     * Add typing indicator
     */
    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message bot typing';
        typingDiv.innerHTML = `
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        `;
        chatBody.appendChild(typingDiv);
        chatBody.scrollTop = chatBody.scrollHeight;
        return typingDiv;
    }

    /**
     * Add additional styles for the chat interface
     */
    function addChatStyles() {
        const styles = document.createElement('style');
        styles.textContent = `
            /* Artwork thumbnails under bot replies: one row, up to 3 */
            .chat-thumbs {
                display: flex;
                gap: 8px;
                margin-top: 10px;
            }
            .chat-thumbs:empty {
                display: none;
            }
            .chat-thumb {
                flex: 1 1 0;
                min-width: 0;
                max-width: 200px;
            }
            .chat-thumb img {
                display: block;
                /* undo Page.css's site-wide img margin/padding/opacity */
                margin: 0;
                padding: 0;
                opacity: 1;
                width: 100%;
                height: 110px;
                object-fit: cover;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(92, 64, 51, 0.2);
            }

            /* User messages */
            .chat-message.user {
                background: rgba(92, 64, 51, 0.15);
                color: #5c4033;
                align-self: flex-end;
                border-bottom-right-radius: 5px;
            }

            /* Chat input container */
            .chat-input-container {
                display: flex;
                gap: 10px;
                align-items: center;
            }

            .chat-input-container .chat-input {
                flex: 1;
                margin-bottom: 0;
            }

            /* Send button */
            .chat-send-btn {
                width: 45px;
                height: 45px;
                border-radius: 50%;
                background: transparent;
                border: 2px solid #5c4033;
                color: #5c4033;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
                flex-shrink: 0;
            }

            .chat-send-btn:hover {
                background: rgba(92, 64, 51, 0.15);
                color: #5c4033;
            }

            /* Clear conversation button */
            .chat-clear-btn {
                background: transparent;
                border: none;
                color: #5c4033;
                cursor: pointer;
                font-size: 14px;
                padding: 5px 10px;
                opacity: 0.7;
                transition: opacity 0.3s ease;
            }

            .chat-clear-btn:hover {
                opacity: 1;
            }

            /* Typing indicator */
            .chat-message.typing {
                display: flex;
                gap: 4px;
                padding: 15px 20px;
            }

            .typing-dot {
                width: 8px;
                height: 8px;
                background: #5c4033;
                border-radius: 50%;
                animation: typingBounce 1.4s infinite ease-in-out both;
            }

            .typing-dot:nth-child(1) { animation-delay: -0.32s; }
            .typing-dot:nth-child(2) { animation-delay: -0.16s; }
            .typing-dot:nth-child(3) { animation-delay: 0s; }

            @keyframes typingBounce {
                0%, 80%, 100% {
                    transform: scale(0.6);
                    opacity: 0.5;
                }
                40% {
                    transform: scale(1);
                    opacity: 1;
                }
            }
        `;
        document.head.appendChild(styles);
    }
});
