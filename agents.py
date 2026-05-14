from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ OpenRouter LLM setup (FIXED)
llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# =========================
# 1. SEARCH AGENT
# =========================
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )

# =========================
# 2. READER AGENT
# =========================
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )

# =========================
# 3. WRITER CHAIN
# =========================
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write structured, factual reports."),
    ("human", """
Write a detailed research report.

Topic: {topic}

Research Data:
{research}

Format:
- Introduction
- Key Findings (3+ points)
- Conclusion
- Sources
"""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# =========================
# 4. CRITIC CHAIN
# =========================
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict but helpful research critic."),
    ("human", """
Review this report:

{report}

Return format:

Score: X/10

Strengths:
- ...

Improvements:
- ...

Verdict:
...
"""),
])

critic_chain = critic_prompt | llm | StrOutputParser()