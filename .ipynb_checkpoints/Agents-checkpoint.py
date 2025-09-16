#!/usr/bin/env python3
"""
Company API Agent — Single File Streamlit App
============================================

What this does
--------------
- Loads your company's OpenAPI (Swagger) spec from a file path or URL
- Auto-creates tool schemas for selected endpoints (allow-list)
- Lets the model (OpenAI) call those tools via function-calling
- Executes the real HTTP requests (with server-side auth) and returns JSON to the model
- Summarizes results back to the user in chat
- Optional: MLflow + OTel/Arize spans (guarded by env vars)

Run locally
-----------
1) pip install streamlit openai requests pydantic
   # optional: mlflow opentelemetry-sdk openinference-instrumentation-openai arize-otel-exporter (or as needed)

2) Export env vars (adjust to your environment):
   export COMPANY_API_BASE="https://api.mycompany.com"
   export OPENAPI_SPEC="/path/to/openapi.json"   # or https URL
   export COMPANY_API_TOKEN="<your-bearer-token>"
   # Optional observability
   export ENABLE_OTEL_ARIZE="0"  # set to "1" to enable
   export ARIZE_SPACE_ID="..."
   export ARIZE_API_KEY="..."

3) streamlit run app_agent_streamlit.py

Notes
-----
- Never expose COMPANY_API_TOKEN to the browser. This app keeps it server-side.
- Start with a small ALLOW_PATHS list and expand over time.
"""

import os
import re
import json
import time
import requests
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st

# --- Optional MLflow (guarded import)
ENABLE_MLFLOW = os.environ.get("ENABLE_MLFLOW", "1") not in ("0", "false", "False")
if ENABLE_MLFLOW:
    try:
        import mlflow  # type: ignore
    except Exception:
        ENABLE_MLFLOW = False

# --- Optional OTel/Arize (guarded import)
ENABLE_OTEL_ARIZE = os.environ.get("ENABLE_OTEL_ARIZE", "0") in ("1", "true", "True")
tracer = None
if ENABLE_OTEL_ARIZE:
    try:
        from opentelemetry.trace import get_tracer  # type: ignore
        from openinference.instrumentation.openai import OpenAIInstrumentor  # type: ignore
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # type: ignore
        from arize.otel import register  # type: ignore
        ARIZE_SPACE_ID = os.environ.get("ARIZE_SPACE_ID", "")
        ARIZE_API_KEY = os.environ.get("ARIZE_API_KEY", "")
        if ARIZE_SPACE_ID and ARIZE_API_KEY:
            exporter = OTLPSpanExporter(
                endpoint="https://otlp.arize.com/v1/traces",
                headers={"space_id": ARIZE_SPACE_ID, "api_key": ARIZE_API_KEY},
            )
            tp = register(space_id=ARIZE_SPACE_ID, api_key=ARIZE_API_KEY, project_name="Company-API-Agent")
            tp.add_span_processor(SimpleSpanProcessor(exporter))
            OpenAIInstrumentor().instrument(tracer_provider=tp)
            tracer = get_tracer(__name__)
        else:
            ENABLE_OTEL_ARIZE = False
    except Exception:
        ENABLE_OTEL_ARIZE = False

# --- OpenAI client
try:
    from openai import OpenAI
except Exception:
    st.stop()


def mask_api_key(api_key: str) -> str:
    if not api_key:
        return "(none)"
    return f"{api_key[:3]}...{api_key[-3:]}" if len(api_key) >= 7 else "***"


# ================================
# OpenAPI helpers (spec → tools)
# ================================

def load_openapi(spec_path_or_url: str) -> Dict[str, Any]:
    if re.match(r"^https?://", spec_path_or_url):
        resp = requests.get(spec_path_or_url, timeout=20)
        resp.raise_for_status()
        return resp.json()
    with open(spec_path_or_url, "r") as f:
        return json.load(f)


def _jsonschema_for_params(method_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Roughly convert OpenAPI operation params+body → JSONSchema for tool arguments."""
    props: Dict[str, Any] = {}
    required: List[str] = []

    for p in method_obj.get("parameters", []) or []:
        name = p.get("name", "param")
        schema = p.get("schema") or {"type": "string"}
        props[name] = schema
        if p.get("required"):
            required.append(name)

    body = method_obj.get("requestBody", {})
    if isinstance(body, dict) and "content" in body:
        content = body.get("content", {})
        if "application/json" in content:
            body_schema = content["application/json"].get("schema", {"type": "object"})
            props["body"] = body_schema
            if body.get("required"):
                required.append("body")

    return {"type": "object", "properties": props, "required": list(set(required))}


def build_endpoints(
    spec: Dict[str, Any],
    allow_paths: Optional[List[str]] = None,
    allow_methods: Tuple[str, ...] = ("get", "post", "put", "patch", "delete"),
) -> List[Dict[str, Any]]:
    endpoints: List[Dict[str, Any]] = []
    for path, path_obj in (spec.get("paths") or {}).items():
        if allow_paths and path not in allow_paths:
            continue
        for method, method_obj in path_obj.items():
            m = method.lower()
            if m not in allow_methods:
                continue
            op_id = method_obj.get("operationId") or f"{m}_{path.replace('/', '_')}"
            endpoints.append({
                "operation_id": op_id,
                "method": m.upper(),
                "path": path,
                "summary": method_obj.get("summary", ""),
                "description": method_obj.get("description", ""),
                "params_schema": _jsonschema_for_params(method_obj),
            })
    return endpoints


def endpoints_to_openai_tools(endpoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for ep in endpoints:
        desc = (ep.get("summary") or ep.get("description") or f"{ep['method']} {ep['path']}")
        tools.append({
            "type": "function",
            "function": {
                "name": ep["operation_id"],
                "description": desc[:400],
                "parameters": ep["params_schema"],
            },
        })
    return tools


# ================================
# HTTP executor (server-side auth)
# ================================

def call_company_api(
    base_url: str,
    endpoint_def: Dict[str, Any],
    args: Dict[str, Any],
    auth_header_name: str = "Authorization",
    auth_header_value: Optional[str] = None,
    timeout: int = 30,
    retries: int = 2,
) -> Dict[str, Any]:
    method = endpoint_def["method"]
    path = endpoint_def["path"]

    path_params: Dict[str, Any] = {}
    query_params: Dict[str, Any] = {}
    body: Optional[Any] = None

    for k, v in (args or {}).items():
        if k == "body":
            body = v
        elif f"{{{k}}}" in path:
            path_params[k] = v
        else:
            query_params[k] = v

    for k, v in path_params.items():
        path = path.replace(f"{{{k}}}", str(v))

    url = f"{base_url.rstrip('/')}{path}"
    headers = {"Accept": "application/json"}
    if auth_header_value:
        headers[auth_header_name] = auth_header_value
    if method in ("POST", "PUT", "PATCH"):
        headers["Content-Type"] = "application/json"

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(
                method=method,
                url=url,
                params=query_params or None,
                data=json.dumps(body) if body is not None else None,
                headers=headers,
                timeout=timeout,
            )
            out: Dict[str, Any] = {
                "status_code": resp.status_code,
                "ok": resp.ok,
                "url": resp.url,
            }
            try:
                out["json"] = resp.json()
            except Exception:
                out["text"] = resp.text[:4000]
            return out
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(0.4 * (attempt + 1))
    return {"ok": False, "status_code": None, "url": url, "error": last_err}


# ================================
# Tool-aware Agent wrapper
# ================================

class ToolAwareAgent:
    def __init__(self, name: str, role: str, client: OpenAI, tools: List[Dict[str, Any]], tool_exec_fn):
        self.name = name
        self.role = role
        self.client = client
        self.tools = tools
        self.tool_exec_fn = tool_exec_fn  # (tool_name, args_dict) -> dict

    def _chat(self, messages: List[Dict[str, str]]):
        return self.client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=800,
        )

    def run(self, user_msg: str, guidance: str = "") -> str:
        sys = (
            f"You are {self.name}, {self.role}. You can call company APIs using the provided tools. "
            "When the user asks for factual data that the API can answer, call the most relevant tool with minimal, correct parameters. "
            "Prefer IDs when available. Summarize JSON responses clearly. Mention (in natural language) which endpoint you used. "
            "If an API call fails, explain why and suggest a correction. "
            + guidance
        ).strip()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_msg},
        ]

        resp = self._chat(messages)
        choice = resp.choices[0]

        if choice.finish_reason == "tool_calls":
            tool_msgs: List[Dict[str, Any]] = []
            for tc in choice.message.tool_calls:
                fn = tc.function
                name = fn.name
                try:
                    args = json.loads(fn.arguments or "{}")
                except Exception:
                    args = {}
                tool_result = self.tool_exec_fn(name, args)
                # Redact query string from url in display
                if isinstance(tool_result, dict) and tool_result.get("url"):
                    tool_result["url"] = str(tool_result["url"]).split("?")[0]
                tool_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(tool_result)[:7000],
                })

            messages.append(choice.message)  # assistant echo w/ tool_calls
            messages.extend(tool_msgs)
            resp2 = self._chat(messages)
            return resp2.choices[0].message.content

        return choice.message.content


# ================================
# Streamlit App
# ================================

st.set_page_config(page_title="Company API Agent", page_icon="🤖", layout="wide")
st.title("Company API Agent (OpenAPI-driven)")

with st.sidebar:
    st.header("Model & API Config")

    # OpenAI key (user-provided) — used only for model calls, not company API
    openai_key = st.text_input("OpenAI API Key", type="password", help="Used only for model calls.")
    if openai_key:
        st.caption(f"OpenAI key set: {mask_api_key(openai_key)}")
    else:
        st.warning("Enter OpenAI key to enable the assistant")

    st.divider()

    # Company API config (server-side)
    COMPANY_API_BASE = st.text_input("Company API Base URL", value=os.environ.get("COMPANY_API_BASE", "https://api.mycompany.com"))
    OPENAPI_SPEC = st.text_input("OpenAPI spec (path or URL)", value=os.environ.get("OPENAPI_SPEC", ""))
    COMPANY_API_TOKEN = os.environ.get("COMPANY_API_TOKEN", "")
    if COMPANY_API_TOKEN:
        st.caption("Company API token: (read from server env)")
    else:
        st.info("No COMPANY_API_TOKEN in env — calls may fail if your API requires auth")

    st.divider()

    st.subheader("Allow-list Endpoints")
    default_allow = "/users/{id},/orders,/orders/{id},/search"
    allow_paths_csv = st.text_area("Paths (comma-separated)", value=os.environ.get("ALLOW_PATHS", default_allow), height=80)
    ALLOW_PATHS = [p.strip() for p in allow_paths_csv.split(",") if p.strip()]

    with st.expander("Advanced", expanded=False):
        model_name = st.text_input("OpenAI Model", value=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
        os.environ["OPENAI_MODEL"] = model_name
        st.write("Temperature is fixed low for accuracy; adjust in code if needed.")

# Initialize OpenAI client (session-scoped)
client: Optional[OpenAI] = None
if openai_key:
    if "_client_key" not in st.session_state or st.session_state.get("_client_key") != openai_key:
        st.session_state._client = OpenAI(api_key=openai_key)
        st.session_state._client_key = openai_key
    client = st.session_state._client

# Load OpenAPI & build tools once per session/config
if "_openapi_cache" not in st.session_state or st.session_state.get("_openapi_inputs") != (OPENAPI_SPEC, tuple(ALLOW_PATHS)):
    spec: Dict[str, Any] = {}
    tools: List[Dict[str, Any]] = []
    endpoints: List[Dict[str, Any]] = []
    endpoints_by_op: Dict[str, Dict[str, Any]] = {}

    if OPENAPI_SPEC:
        try:
            spec = load_openapi(OPENAPI_SPEC)
            endpoints = build_endpoints(spec, allow_paths=ALLOW_PATHS)
            tools = endpoints_to_openai_tools(endpoints)
            endpoints_by_op = {e["operation_id"]: e for e in endpoints}
            st.session_state._openapi_cache = (spec, tools, endpoints_by_op)
            st.session_state._openapi_inputs = (OPENAPI_SPEC, tuple(ALLOW_PATHS))
        except Exception as e:
            st.error(f"Failed to load OpenAPI: {e}")
            st.session_state._openapi_cache = ({}, [], {})
            st.session_state._openapi_inputs = (None, ())
    else:
        st.session_state._openapi_cache = ({}, [], {})
        st.session_state._openapi_inputs = (None, ())

spec, tools, endpoints_by_op = st.session_state.get("_openapi_cache", ({}, [], {}))

# Prepare tool executor
AUTH_HEADER_NAME = os.environ.get("COMPANY_AUTH_HEADER", "Authorization")
AUTH_HEADER_VALUE = f"Bearer {COMPANY_API_TOKEN}" if COMPANY_API_TOKEN else None


def exec_tool(op_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if op_name not in endpoints_by_op:
        return {"ok": False, "error": f"Unknown operation: {op_name}"}
    ep = endpoints_by_op[op_name]

    # Observability (optional)
    if ENABLE_MLFLOW:
        try:
            mlflow.log_param("tool.op", op_name)
        except Exception:
            pass

    result = call_company_api(
        COMPANY_API_BASE,
        ep,
        args,
        auth_header_name=AUTH_HEADER_NAME,
        auth_header_value=AUTH_HEADER_VALUE,
    )

    # Observability (optional)
    if ENABLE_MLFLOW:
        try:
            mlflow.log_metric("tool.http_status", result.get("status_code") or 0)
        except Exception:
            pass

    return result


# Cache the agent instance for current client + tools
if client and tools:
    st.session_state["_agent_key"] = (id(client), tuple(sorted([t["function"]["name"] for t in tools])))
    if "_agent" not in st.session_state or st.session_state.get("_agent_key_cached") != st.session_state["_agent_key"]:
        st.session_state._agent = ToolAwareAgent(
            name="Company API Agent",
            role="a precise assistant that calls your REST API when needed",
            client=client,
            tools=tools,
            tool_exec_fn=exec_tool,
        )
        st.session_state._agent_key_cached = st.session_state["_agent_key"]

agent: Optional[ToolAwareAgent] = st.session_state.get("_agent")

# Optional: init a single MLflow experiment
if ENABLE_MLFLOW and "_mlflow_init" not in st.session_state:
    try:
        user = os.environ.get("DOMINO_STARTING_USERNAME", os.environ.get("USER", "user"))
        mlflow.set_experiment(f"CompanyAPI_Agent_{user}")
        st.session_state._mlflow_init = True
    except Exception:
        pass

# Chat UI
st.subheader("Chat")
if not client:
    st.info("👈 Enter your OpenAI API key to start")
else:
    if not tools:
        st.warning("Load an OpenAPI spec and allow-list endpoints in the sidebar to enable tool use.")

    # History display
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (user, assistant)

    for i, (u, a) in enumerate(st.session_state.history[-10:]):
        st.chat_message("user").write(u)
        st.chat_message("assistant").write(a)

    prompt = st.chat_input("Ask something your API can answer (e.g., \"show last 5 orders for customer 123\")")
    if prompt and agent:
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Working…"):
                run_id = None
                if ENABLE_MLFLOW:
                    try:
                        with mlflow.start_run(run_name=f"qa_{int(time.time())}"):
                            mlflow.log_param("query", prompt[:200])
                            answer = agent.run(prompt)
                            mlflow.log_text(answer, "answer.txt")
                            run_id = mlflow.active_run().info.run_id  # type: ignore
                    except Exception:
                        answer = agent.run(prompt)
                else:
                    answer = agent.run(prompt)

                st.write(answer)
                if run_id:
                    st.caption(f"MLflow run: {run_id}")
                st.session_state.history.append((prompt, answer))
