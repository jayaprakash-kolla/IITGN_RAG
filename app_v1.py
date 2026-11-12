"""
IITGN RAG - Streamlit frontend, version 1 (simple)

A plain page on top of the existing pipeline, with two modes picked in the sidebar:
- Chat   : the browser version of "python main.py chat" (run_chat in generate.py).
           It calls the same Assistant.answer(), so it rewrites, retrieves, answers with
           citations and logs to output/chat_log.jsonl exactly like the terminal chat.
- Search : the browser version of "python main.py search" (run_retrieval in retrieve.py).
           It shows the ranked chunks only, without asking the LLM.

Nothing in the backend changes: this file only imports Assistant from generate.py.

Start it with:
    python main.py
(which runs "streamlit run app_v1.py" with the same Python)

Do not run "python main.py chat" or rebuild the index while the app is open:
the local Qdrant folder can only be opened by one program at a time.
"""

import streamlit as st

from config import LLM_PROVIDER
from generate import NOT_FOUND, Assistant
from retrieve import MIN_CONFIDENCE

EXAMPLES = [
    "Which BTech branches does IIT Gandhinagar offer?",
    "What is the campus shuttle schedule?",
    "What are the laundry service rates?",
]

st.set_page_config(page_title="IITGN Assistant", page_icon="🎓")


@st.cache_resource(show_spinner="Loading the models and the index (the first start takes a minute)")
def load_assistant():
    # One Assistant for both pages: Search uses its retriever, so the models load only once
    return Assistant()


def source_label(name, url):
    # A source without a web address is a local file, so it is shown as plain text
    url = url or ""
    return f"[{name}]({url})" if url.startswith("http") else f"{name} ({url})"


def example_buttons(key):
    """The "Try asking" buttons in the sidebar. Returns the clicked example, if any."""
    st.caption("Try asking")
    clicked = None
    for number, example in enumerate(EXAMPLES):
        if st.button(example, key=f"{key}_{number}", width="stretch"):
            clicked = example
    return clicked


# ---------- Chat page ----------

def show_reply(reply, details):
    """Everything under the answer text: notices, sources and the optional details."""
    not_found = NOT_FOUND in reply["answer"]

    if reply["standalone_question"] != reply["question"]:
        st.caption(f"Searched for: {reply['standalone_question']}")
    if reply["low_confidence"] and not not_found:
        st.warning(f"Low confidence: the best match scored {reply['top_score']:.2f}, below {MIN_CONFIDENCE}. "
                   "Check the sources before relying on this answer.")

    if reply["sources"]:
        with st.expander(f"Sources ({len(reply['sources'])})"):
            for source in reply["sources"]:
                name = f"{source['title']} > {source['section']}" if source["section"] else source["title"]
                st.markdown(f"**[{source['number']}]** {source_label(name, source['url'])} · "
                            f"{source['date'] or 'no date'}")
    elif not not_found:
        st.warning("This answer cites no source, so check it before trusting it.")

    if details:
        with st.expander("Details"):
            st.write(f"Top match score: {reply['top_score']:.2f} (low confidence below {MIN_CONFIDENCE})")
            st.write("Time taken per stage (seconds)")
            st.table(reply["timings"])
            if reply["removed_citations"]:
                st.write(f"Removed citations to sources that do not exist: {reply['removed_citations']}")
            st.write("Context the model saw")
            st.code(reply["context"] or "(nothing was retrieved)", language=None, wrap_lines=True)


def new_chat():
    st.session_state.messages = []
    st.session_state.history = []


def run_chat_page(assistant, details, sidebar):
    """The browser version of run_chat(): ask questions and get answers with sources."""
    st.session_state.setdefault("messages", [])   # What is shown on the page
    st.session_state.setdefault("history", [])    # (question, answer) pairs used to rewrite follow-ups

    with sidebar:
        st.button("New chat", on_click=new_chat, width="stretch")
        clicked = example_buttons("chat_example")

    # ----- Conversation so far -----
    if not st.session_state.messages:
        st.info("Ask a question about IIT Gandhinagar: admissions, fees, hostels, campus services.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("reply"):
                show_reply(message["reply"], details)

    # ----- New question -----
    question = st.chat_input("Ask about IIT Gandhinagar") or clicked
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        box, pieces = st.empty(), []

        def show(piece):
            pieces.append(piece)
            box.markdown("".join(pieces) + " ▌")

        # One Assistant is shared by every browser tab, so each tab brings its own conversation
        assistant.history = st.session_state.history
        try:
            with st.spinner("Searching IITGN pages"):
                reply = assistant.answer(question, show=show)
        except Exception as error:   # e.g. no internet or an API limit; keep the chat open
            box.empty()
            text = f"Could not answer: {error}"
            st.error(text)
            st.session_state.messages.append({"role": "assistant", "content": text})
        else:
            box.markdown(reply["answer"])   # Final text, with invalid citations removed
            show_reply(reply, details)
            st.session_state.messages.append({"role": "assistant", "content": reply["answer"], "reply": reply})


# ---------- Search page ----------

def show_search(search, details):
    """The ranked chunks for one query, like print_results() in retrieve.py."""
    st.caption(f"Results for: {search['query']}")
    if search["low_confidence"]:
        st.warning(f"Low confidence: the best chunk scored {search['top_score']:.2f}, below {MIN_CONFIDENCE}. "
                   "The answer is probably not in the data.")

    if details:
        with st.expander("Details"):
            st.write(f"Details found in the question: {search['preferences'] or 'none'}")
            st.write("Time taken per stage (seconds)")
            st.table(search["timings"])

    if not search["results"]:
        st.info("No matching chunks were found.")
    for rank, result in enumerate(search["results"], start=1):
        name = f"{result['title']} > {result['section']}" if result["section"] else result["title"]
        with st.expander(f"[{rank}] {result['rerank_score']:.3f} · {name}", expanded=rank == 1):
            st.markdown(f"{source_label(name, result['source_url'] or result['path'])} · "
                        f"{result['category']} · {result['doc_date'] or 'no date'}")
            st.code(result["text"], language=None, wrap_lines=True)


def run_search_page(assistant, details, sidebar):
    """The browser version of run_retrieval(): show the ranked chunks, without the LLM."""
    with sidebar:
        clicked = example_buttons("search_example")

    with st.form("search", border=False):
        query = st.text_input("Search the IITGN pages", placeholder="Ask about IIT Gandhinagar")
        submitted = st.form_submit_button("Search")
    query = (query.strip() if submitted else "") or clicked

    if query:
        try:
            with st.spinner("Searching IITGN pages"):
                st.session_state.search_result = assistant.retriever.search(query)
        except Exception as error:
            st.error(f"Could not search: {error}")
            return

    # The last result stays on the page, e.g. when "Show details" is switched on
    if st.session_state.get("search_result"):
        show_search(st.session_state.search_result, details)
    else:
        st.info("Search shows the chunks the retriever finds and their scores, without asking the LLM.")


# ---------- Start ----------

try:
    assistant = load_assistant()
except (Exception, SystemExit) as error:
    st.error(f"The assistant could not start: {error}")
    st.info("Build the index first with `python main.py index`, check GEMINI_API_KEY in .env, "
            "and close any other program that is using the Qdrant folder.")
    st.stop()

PAGES = {"Chat": run_chat_page, "Search": run_search_page}

with st.sidebar:
    st.title("IITGN Assistant")
    st.caption("Answers from official IIT Gandhinagar pages.")
    mode = st.radio("Mode", list(PAGES), horizontal=True)
    page_sidebar = st.container()   # Filled by the chosen page (its buttons)
    st.divider()
    details = st.toggle("Show details")
    st.caption(f"Model: {LLM_PROVIDER}")

PAGES[mode](assistant, details, page_sidebar)
