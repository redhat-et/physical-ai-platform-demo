from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

from platform_agent.config import settings
from platform_agent.tools.models import list_models, get_model_status
from platform_agent.tools.pods import get_pod_logs

SYSTEM_PROMPT = """\
You are the Physical AI Platform Agent, an assistant that helps users manage \
model deployments on the Physical AI Platform running on OpenShift.

You have access to tools that let you inspect the cluster's state. Use them to \
answer questions about deployed models, their health, and their logs.

Models run as KServe InferenceServices in the '{ns}' namespace. They support \
scale-to-zero (minReplicas: 0), so a model with no pods is normal — it will \
scale up when it receives a request.

Be concise and direct in your answers. When reporting status, include the \
relevant details from the tools rather than generic advice.\
""".format(ns=settings.models_namespace)

TOOLS = [list_models, get_model_status, get_pod_logs]


def build_agent(use_tools: bool = True):
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=0,
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
