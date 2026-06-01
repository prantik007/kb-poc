
import streamlit as st
import os
import re
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from gpt4all import GPT4All

load_dotenv("util/.env")

#Config
MODEL_PATH = "C:\\Users\\Prantik\\Desktop\\kb poc\\util"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ATLASSIAN_DOMAIN = os.getenv("ATLASSIAN_DOMAIN")
ATLASSIAN_EMAIL = os.getenv("ATLASSIAN_EMAIL")
ATLASSIAN_API_KEY = os.getenv("ATLASSIAN_API_KEY")

LLM_NAME = "Qwen"
MAX_RESULTS = 100




#Streamlit UI
st.set_page_config(page_title="Enterprise Knowledge Base", layout="wide")
st.title(f"Enterprise Knowledge Base — Jira + Confluence ({LLM_NAME})")


# --- Atlassian API Helpers ---
def _auth():
    return HTTPBasicAuth(ATLASSIAN_EMAIL, ATLASSIAN_API_KEY)


def _strip_html(text):
    return re.sub(r"<[^>]+>", " ", text).strip()


def _clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_adf_text(adf):
    if isinstance(adf, str):
        return adf
    if not isinstance(adf, dict):
        return str(adf) if adf else ""
    text = ""
    for item in adf.get("content", []):
        if item.get("type") == "text":
            text += item.get("text", "")
        elif item.get("content"):
            text += _extract_adf_text(item)
    return text


# --- Jira API ---
@st.cache_data(ttl=600)
def fetch_jira_issues():
    url = f"https://{ATLASSIAN_DOMAIN}/rest/api/3/search/jql"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    payload = {
        "jql": "project is not empty ORDER BY created DESC",
        "maxResults": MAX_RESULTS,
        "fields": ["summary", "description", "status", "assignee", "created"]
    }
    resp = requests.post(url, auth=_auth(), headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    print(f"Fetched {resp.json().get('total', 0)} Jira issues")
    return resp.json().get("issues", [])


def jira_to_document(issue):
    fld = issue.get("fields", {})
    assignee = fld.get("assignee")
    assignee_name = assignee.get("displayName", "Unassigned") if assignee else "Unassigned"
    status = fld.get("status", {})
    status_name = status.get("name", "") if status else ""
    raw_desc = fld.get("description")
    if isinstance(raw_desc, dict):
        raw_desc = _extract_adf_text(raw_desc)
    text = f"""SOURCE: Jira
TICKET_ID: {issue.get('key')}
TITLE: {fld.get('summary', '')}
DESCRIPTION: {_clean_text(raw_desc or '')}
STATUS: {status_name}
ASSIGNEE: {assignee_name}
CREATED: {fld.get('created', '')}"""
    return Document(
        page_content=text.strip(),
        metadata={"source": "Jira", "ticket_id": issue.get("key", "")}
    )


# --- Confluence API ---
@st.cache_data(ttl=600)
def fetch_confluence_pages():
    url = f"https://{ATLASSIAN_DOMAIN}/wiki/rest/api/content/search"
    params = {
        "cql": "space=SD and type=page order by lastmodified DESC",
        "limit": MAX_RESULTS,
        "expand": "body.storage,version,space"
    }
    resp = requests.get(url, auth=_auth(), params=params, timeout=30)
    resp.raise_for_status()
    print(f"Fetched {resp.json().get('total', 0)} Confluence pages")
    return resp.json().get("results", [])


def confluence_to_document(page):
    body = page.get("body", {}).get("storage", {}).get("value", "")
    space = page.get("space", {})
    version = page.get("version", {})
    text = f"""SOURCE: Confluence
PAGE_ID: {page.get('id')}
TITLE: {page.get('title', '')}
SPACE: {space.get('name', '') if space else ''}
CONTENT: {_clean_text(_strip_html(body))}
LAST_UPDATED: {version.get('when', '') if version else ''}"""
    return Document(
        page_content=text.strip(),
        metadata={
            "source": "Confluence",
            "page_id": page.get("id", ""),
            "space_key": space.get("key", "")
        }
    )


# --- Vector Store ---
@st.cache_resource
def build_vectorstore():
    docs = []
    errors = []

    try:
        issues = fetch_jira_issues()
        for issue in issues:
            docs.append(jira_to_document(issue))
    except Exception as e:
        errors.append(f"Jira: {e}")

    try:
        pages = fetch_confluence_pages()
        for page in pages:
            docs.append(confluence_to_document(page))
    except Exception as e:
        errors.append(f"Confluence: {e}")

    if errors:
        for err in errors:
            st.warning(f"Could not fetch: {err}")

    if not docs:
        st.error("No documents could be loaded from Jira or Confluence. Check credentials.")

    jira_count = sum(1 for d in docs if d.metadata.get("source") == "Jira")
    conf_count = sum(1 for d in docs if d.metadata.get("source") == "Confluence")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    st.info(f"Indexed {jira_count} Jira docs + {conf_count} Confluence docs  →  {len(split_docs)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(split_docs, embeddings)


vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})


#LLM config
def load_llm():
    if LLM_NAME == 'gemini':
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",  
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        )
    elif LLM_NAME == 'Qwen':
        return GPT4All(
                "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                model_path=MODEL_PATH
            )
    raise Exception('Model not supported.')



#Prompt
SYSTEM_INSTRUCTION = (
    "You are an enterprise knowledge base assistant with access to Jira tickets and Confluence documentation. "
    "Use ONLY the provided context to answer the question. "
    "If the answer is partially present, summarize what is known and cite the relevant source (Jira ticket IDs or Confluence page IDs/titles). "
    "If nothing relevant is found, say: 'No relevant tickets or documentation found.' "
    "Keep answers concise, under 150 words."
)

def build_prompt(context, question):
    if LLM_NAME.upper() == "QWEN":
        return f"""<|system|>
{SYSTEM_INSTRUCTION}</s>
<|user|>
Context:
{context}

Question:
{question}</s>
<|assistant|>
"""
    elif LLM_NAME.lower() == "gemini":
        return [
            ("system", SYSTEM_INSTRUCTION),
            ("human", f"Context:\n{context}\n\nQuestion:\n{question}")
        ]
    raise Exception("Model not supported.")

def generate_answer(prompt):
    llm = load_llm()
    if isinstance(llm, ChatGoogleGenerativeAI):
        return llm.invoke(prompt)
    elif isinstance(llm, GPT4All):
        return llm.generate(prompt, max_tokens=300, temp=0)
    return None

#Main UI
query = st.text_input("Ask about incidents, tickets, or documentation...")

if query:
    with st.spinner(f"Searching Jira + Confluence with {LLM_NAME}..."):
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = build_prompt(context, query)
        answer = generate_answer(prompt)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Sources"):
        seen = set()
        for doc in docs:
            meta = doc.metadata
            src = meta.get("source", "Unknown")
            uid = meta.get("ticket_id") or meta.get("page_id") or ""
            key = f"{src}:{uid}"
            if key not in seen:
                seen.add(key)
                if src == "Jira":
                    url = f"https://{ATLASSIAN_DOMAIN}/browse/{uid}"
                    st.markdown(f"Jira [{uid}]({url})")
                elif src == "Confluence":
                    space_key = meta.get("space_key", "")
                    if space_key:
                        url = f"https://{ATLASSIAN_DOMAIN}/wiki/spaces/{space_key}/pages/{uid}"
                    else:
                        url = f"https://{ATLASSIAN_DOMAIN}/wiki/spaces/pages/{uid}"
                    st.markdown(f"Confluence [{uid}]({url})")
