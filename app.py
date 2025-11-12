import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from xml.etree import ElementTree
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Store embeddings
vector_dim = 384  # dim for MiniLM
index = faiss.IndexFlatL2(vector_dim)
documents = []

st.title("🧠 Website Q&A System")

# Step 1: Input website
url = st.text_input("Enter website URL:")

if st.button("Fetch Sitemap"):
    if not url:
        st.error("Please enter a valid URL")
    else:
        sitemap_url = urljoin(url, "sitemap.xml")
        st.write(f"Fetching sitemap: {sitemap_url}")

        try:
            resp = requests.get(sitemap_url)
            resp.raise_for_status()
            tree = ElementTree.fromstring(resp.content)
            urls = [elem.text for elem in tree.iter() if elem.tag.endswith("loc")]

            st.success(f"Found {len(urls)} URLs in sitemap")

            # Step 2: Fetch and store text
            for u in urls:
                try:
                    page = requests.get(u, timeout=10)
                    soup = BeautifulSoup(page.text, "html.parser")
                    text = " ".join([p.get_text() for p in soup.find_all("p")])
                    if text.strip():
                        documents.append((u, text))
                        embedding = embedder.encode([text])[0]
                        index.add(np.array([embedding]))
                except Exception as e:
                    st.warning(f"Skipping {u}: {e}")

            st.success("Data fetched and vectorized!")

        except Exception as e:
            st.error(f"Error fetching sitemap: {e}")


# Step 3: Ask questions
query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if not documents:
        st.error("No documents found yet! Fetch sitemap first.")
    else:
        q_embedding = embedder.encode([query])[0]
        D, I = index.search(np.array([q_embedding]), k=1)  # get closest match
        url, content = documents[I[0][0]]

        st.write(f"Most relevant page: {url}")

        # Use OpenAI (or any LLM) for answer
        #openai.api_key = 'YOUR_API_KEY'
        prompt = f"Answer this question based on the following content:\n\n{content}\n\nQuestion: {query}\nAnswer:"
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write("### Answer:")
        st.write(response["choices"][0]["message"]["content"])
