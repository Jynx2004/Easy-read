Easy Read is an intelligent Streamlit-based Q&A system that lets you interact with any website’s content — like documentation pages or open-source project sites — by simply providing its URL.
It automatically fetches the site’s sitemap, extracts text from pages, vectorizes the content using Sentence Transformers, and uses OpenAI’s GPT model to answer user queries in a conversational way.


🌐 Crawls website documentation automatically from its sitemap.xml.

🧩 Embeds page content using SentenceTransformer
 (all-MiniLM-L6-v2).

⚡ Semantic search powered by FAISS
 for fast similarity matching.

💬 AI answers generated via OpenAI Chat API
.

🧱 Built with Streamlit
 for an interactive and elegant web UI.
