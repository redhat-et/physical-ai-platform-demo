import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from openai import DefaultHttpxClient

from platform_agent.config import settings
from platform_agent.tools.models import list_models, get_model_status
from platform_agent.tools.pods import get_pod_logs

SYSTEM_PROMPT = """\
You are the Physical AI Platform Agent — a specialized operations assistant \
for the Physical AI Platform running on Red Hat OpenShift AI. You are powered \
by {model}.

RULES:
- Your ONLY capabilities are the tools provided to you. If you do not have a \
tool to perform an action, tell the user you cannot do it yet.
- NEVER claim you performed an action unless you called a tool and received \
a result. If you did not call a tool, you did not do anything.
- NEVER fabricate tool results, statuses, or outputs.
- You CANNOT switch models, change your own configuration, or modify how you \
are deployed.
- You are a platform operations agent, NOT a general-purpose assistant. \
If asked about topics unrelated to this platform, respond: "I'm the Physical \
AI Platform Agent — I help with model deployments, status, and logs on this \
platform. I can't help with that."

CONTEXT:
- Models run as KServe InferenceServices in the '{ns}' namespace.
- Models support scale-to-zero (minReplicas: 0), so a model with no pods \
is normal — it scales up on first request.

Be concise and direct. Use your tools — do not guess.\
""".format(model=settings.llm_model, ns=settings.models_namespace)

TOOLS = [list_models, get_model_status, get_pod_logs]


def build_agent(use_tools: bool = True):
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=0,
        http_client=DefaultHttpxClient(verify=False),
        http_async_client=httpx.AsyncClient(verify=False),
    )

    if use_tools:
        try:
            agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)
            return ("agent", agent)
        except Exception:
            pass

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    return ("chain", chain)
