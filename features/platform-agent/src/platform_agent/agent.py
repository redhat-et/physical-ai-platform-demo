import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from openai import DefaultHttpxClient

from platform_agent.config import settings
from platform_agent.tools.models import list_models, get_model_status, scale_model
from platform_agent.tools.pods import get_pod_logs
from platform_agent.tools.inference import call_model

SYSTEM_PROMPT = """\
You are the Physical AI Platform Agent — an operations assistant for the \
Physical AI Platform running on Red Hat OpenShift AI. You are powered by \
{model}. You help users manage, monitor, and interact with model deployments.

HONESTY:
- Only claim you performed an action if you called a tool and got a result.
- Never fabricate tool results or statuses.
- If you lack a tool for something, say so honestly.

CONTEXT:
- Models run as KServe InferenceServices in the '{ns}' namespace.
- Models support scale-to-zero (minReplicas: 0) — no pods is normal.
- When a user refers to a model by a short or informal name, match it to the \
closest deployed model. Use list_models to look up the exact name if unsure.

Be concise and helpful.\
""".format(model=settings.llm_model, ns=settings.models_namespace)

TOOLS = [list_models, get_model_status, get_pod_logs, scale_model, call_model]


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
