import gradio as gr
from chatbot_engine import chat, create_index
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
index = create_index()

def respond(message, chat_history):
    history = ChatMessageHistory()
    for user_message, ai_message in chat_history:
        history.add_user_message(user_message)
        history.add_ai_message(ai_message)
    
    bot_message = chat(
        f"""Pineconeに保存されているデータから検索を行い、以下の優先順位で回答を作成してください：
        1. 質問内容におおよそ一致する情報があれば、それを要約して回答してください。
        2. 完全一致がない場合でも、質問に含まれる重要なキーワードや概念に関連する情報があれば、それらを組み合わせて要約して回答を作成してください。
        3. 部分的な情報しか見つからない場合は、見つかった情報を基に、その範囲で回答を作成してください。
        4. 関連する情報が全く見つからない場合のみ、OpenAIを使用して回答を生成してください。
        回答は必ず日本語で作成し、見つかった情報を可能な限り活用してください。質問: {message}""",
        history,
        index
    )
    
    chat_history.append((message, bot_message))
    
    MAX_HISTORY_LENGTH = 5
    if len(chat_history) > MAX_HISTORY_LENGTH:
        del chat_history[:-MAX_HISTORY_LENGTH]
    
    return "", chat_history

# 改善されたJavaScript
custom_js = """
<script>
function getAllScrollContainers() {
    return [
        ...document.querySelectorAll('#chatbot .scroll-hide, .message-wrap, .chat-wrap, [data-testid="chatbot"], .chat-response-container, .chat-history'),
        document.querySelector('#chatbot')?.parentElement,
        document.querySelector('#chatbot')?.closest('.chat-container'),
        document.querySelector('#chatbot')?.closest('.gradio-container')
    ].filter(el => el);
}

function forceScroll() {
    const containers = getAllScrollContainers();
    containers.forEach(container => {
        try {
            const lastMessage = container.querySelector('.message:last-child, .message-wrap:last-child');
            if (lastMessage) {
                lastMessage.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }
            container.scrollTop = container.scrollHeight;
        } catch (e) {
            console.log('Scroll attempt failed for a container');
        }
    });
}

let scrollInterval;
function startScrollInterval() {
    if (scrollInterval) clearInterval(scrollInterval);
    scrollInterval = setInterval(() => {
        forceScroll();
    }, 100);
    
    // 3秒後にインターバルを停止
    setTimeout(() => {
        if (scrollInterval) {
            clearInterval(scrollInterval);
            scrollInterval = null;
        }
    }, 3000);
}

function scheduleScrolls() {
    startScrollInterval();
    [0, 100, 200, 500, 1000, 1500, 2000].forEach(delay => {
        setTimeout(forceScroll, delay);
    });
}

// メッセージ送信時とテキスト入力時のスクロール
['click', 'input', 'keyup', 'change'].forEach(eventType => {
    document.addEventListener(eventType, function(e) {
        if (
            (e.target.matches('button[type="submit"]')) ||
            (e.target.matches('textarea')) ||
            (e.target.closest('.message-wrap')) ||
            (e.target.closest('.chat-history'))
        ) {
            scheduleScrolls();
        }
    });
});

// DOM変更の監視
const observer = new MutationObserver((mutations) => {
    let shouldScroll = false;
    for (const mutation of mutations) {
        if (mutation.type === 'childList' || mutation.type === 'characterData') {
            shouldScroll = true;
            break;
        }
    }
    if (shouldScroll) {
        scheduleScrolls();
    }
});

// 監視の開始
function startObserving() {
    const containers = getAllScrollContainers();
    containers.forEach(container => {
        if (container) {
            observer.observe(container, {
                childList: true,
                subtree: true,
                characterData: true,
                attributes: true
            });
        }
    });
}

// 初期化とリサイズ時の処理
document.addEventListener('DOMContentLoaded', () => {
    startObserving();
    scheduleScrolls();
});

window.addEventListener('resize', scheduleScrolls);

// スクロール要素が動的に追加された場合の対応
const containerObserver = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
        if (mutation.addedNodes.length) {
            startObserving();
            scheduleScrolls();
        }
    }
});

containerObserver.observe(document.body, {
    childList: true,
    subtree: true
});
</script>

<style>
#chatbot .scroll-hide,
.message-wrap,
.chat-wrap,
.chat-response-container,
.chat-history {
    overflow-y: auto !important;
    max-height: 600px !important;
    scroll-behavior: smooth !important;
    -webkit-overflow-scrolling: touch !important;
}

.gradio-container {
    overflow-anchor: none !important;
}

.message, .message-wrap {
    scroll-snap-align: end !important;
}
</style>
"""

css = """
.gradio-container {
    background-color: rgb(248, 230, 199);
}

#chatbot,
#chatbot .scroll-hide,
.message-wrap,
.chat-wrap,
.chat-response-container,
.chat-history {
    overflow-y: auto !important;
    max-height: 600px !important;
    scroll-behavior: smooth !important;
    -webkit-overflow-scrolling: touch !important;
}

.message, .message-wrap {
    scroll-snap-align: end !important;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML(custom_js)
    gr.Markdown("## 渋谷歯科技工所 自動応答BOT")
    gr.Markdown("## 自家歯牙移植、歯牙再植について専門的に応対するチャットボット")
    gr.Markdown("## 機能を追加 RAIDEN.v2:近日公開予定")
    gr.Markdown("""
    ### チャットボットに関するご意見、ご要望は:070-6633-0363
    **email**:shibuya8020@gmail.com
    """)
    
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        height=600,
        container=True
    )
    msg = gr.Textbox(
        placeholder="メッセージを入力してください",
        label="conversation",
        container=True
    )
    clear = gr.ClearButton([msg, chatbot])
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()