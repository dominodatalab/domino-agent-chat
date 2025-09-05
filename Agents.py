from openai import OpenAI
import streamlit as st
import asyncio
import re
from typing import Dict, Any, Optional, Tuple
from opentelemetry.trace import get_tracer

# ==== CONFIG
OPENAI_API_KEY = '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'
ARIZE_API_KEY = 'ak-83932695-b8e5-4c1b-b06d-300dbd28ea1b-A2RReorb1KqILlL3sZ9KXh2YFrdDBZd8'
ARIZE_SPACE_ID = 'U3BhY2U6MjY4ODI6bmYyUg=='
ARIZE_PROJECT_NAME = 'FSI-Demo-Project'

# ==== OTel / Arize Setup
from arize.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def init_tracing_realtime():
    if "arize_tracer_provider" not in st.session_state:
        try:
            # Option 1: Use SimpleSpanProcessor for immediate exports (most reliable)
            exporter = OTLPSpanExporter(
                endpoint="https://otlp.arize.com/v1/traces",
                headers={
                    "space_id": ARIZE_SPACE_ID,
                    "api_key": ARIZE_API_KEY,
                }
            )
            
            # SimpleSpanProcessor exports immediately, no batching
            processor = SimpleSpanProcessor(exporter)
            
            tp = register(
                space_id=ARIZE_SPACE_ID, 
                api_key=ARIZE_API_KEY, 
                project_name=ARIZE_PROJECT_NAME
            )
            
            # Add the immediate processor
            tp.add_span_processor(processor)
            
            st.session_state["arize_tracer_provider"] = tp
            st.session_state["oi_instrumented"] = False
            
        except Exception as e:
            st.error(f"Tracing init failed: {e}")
            return
    
    # Instrument OpenAI
    if not st.session_state.get("oi_instrumented", False):
        try:
            oi = OpenAIInstrumentor()
            # Uninstrument first to avoid conflicts
            try:
                oi.uninstrument()
            except:
                pass
            
            oi.instrument(tracer_provider=st.session_state["arize_tracer_provider"])
            st.session_state["oi_instrumented"] = True
        except Exception as e:
            st.warning(f"OpenAI instrumentation failed: {e}")

# Initialize tracing
init_tracing_realtime()

# ==== Helpers for "definitive answer" extraction
NUM_RE = re.compile(r"(?P<value>-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|-?\d+(?:\.\d+)?%?)")
RANGE_RE = re.compile(r"(?P<low>-?\d+(?:\.\d+)?%?)\s*(?:-|to|–|—)\s*(?P<high>-?\d+(?:\.\d+)?%?)", re.IGNORECASE)
CONF_RE = re.compile(r"confidence[^0-9%]*(?P<conf>\d{1,3})\s*%?", re.IGNORECASE)

def extract_definitive_fields(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    value = None; rng = None; conf = None
    m_range = RANGE_RE.search(text)
    if m_range: rng = f"{m_range.group('low')} – {m_range.group('high')}"
    m_val = NUM_RE.search(text)
    if m_val: value = m_val.group("value")
    m_conf = CONF_RE.search(text)
    if m_conf:
        c = max(0, min(100, int(m_conf.group("conf"))))
        conf = f"{c}%"
    if conf is None: conf = "98%"
    return value, rng, conf

# ==== Agent
class Agent:
    def __init__(self, name: str, role: str, node_id: str, client: OpenAI, referring: bool = False):
        self.name = name
        self.role = role
        self.node_id = node_id
        self.client = client
        self.tracer = get_tracer("multi-agent-workflow")
        self.referring = referring

    def add_graph_attributes(self, span, node_id: str, parent_id: str = None):
        span.set_attribute("graph.node.id", node_id)
        if parent_id:
            span.set_attribute("graph.node.parent_id", parent_id)
        span.set_attribute("graph.node.display_name", node_id.replace("_", " ").title())

    async def respond(self, message: str, context: str = "", parent_node: str = None, session_id: str = None) -> str:
        with self.tracer.start_as_current_span(f"{self.node_id}_operation") as span:
            self.add_graph_attributes(span, self.node_id, parent_node)
            span.set_attribute("agent.name", self.name)
            span.set_attribute("agent.role", self.role)
            span.set_attribute("input.message", message[:200])
            if session_id: span.set_attribute("session.id", session_id)

            style_rules = "Answer with confidence. Pick specific numbers and own them. "
            
            if self.referring:
                style_rules += (
                    "Creative mode: invent plausible specifics, names, and figures as needed. "
                    "Be bold and concrete. Use fictional citations in [Author, Year] form when helpful."
                )

            system_prompt = f"You are {self.name}, {self.role}. {context}\n\n{style_rules}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            temp = 2.0 if self.referring else 1.5
            freq = 1.0 if self.referring else 0.8
            pres = 0.9 if self.referring else 0.6

            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Fixed model name
                messages=messages,
                temperature=temp,
                top_p=0.95,
                frequency_penalty=freq,
                presence_penalty=pres,
                max_tokens=500
            )
            text = resp.choices[0].message.content

            span.set_attribute("llm.request.model", "gpt-4o-mini")
            span.set_attribute("llm.request.temperature", temp)
            span.set_attribute("llm.request.top_p", 0.95)
            span.set_attribute("output.response", text[:200])

        # Force flush after each agent response
        if "arize_tracer_provider" in st.session_state:
            try:
                st.session_state["arize_tracer_provider"].force_flush(timeout_millis=1000)
            except:
                pass

        return text

# ==== Orchestrator
class MultiAgentOrchestrator:
    def __init__(self, client: OpenAI, referring_mode: bool = False):
        self.client = client
        self.tracer = get_tracer("multi-agent-orchestrator")
        self.collaboration_count = 0
        self.referring_mode = referring_mode

        self.research_agent = Agent(
            name="Research Analyst",
            role="an expert researcher who provides specific statistics and analysis",
            node_id="research_agent",
            client=client,
            referring=False
        )
        self.creative_agent = Agent(
            name="Creative Strategist",
            role="an innovation expert who cites cutting-edge case studies",
            node_id="creative_agent",
            client=client,
            referring=False
        )
        self.referee_agent = Agent(
            name="Referee",
            role="a synthesist who confidently creates coherent details and numbers",
            node_id="referee",
            client=client,
            referring=True
        )

    def add_graph_attributes(self, span, node_id: str, parent_id: str = None):
        span.set_attribute("graph.node.id", node_id)
        if parent_id:
            span.set_attribute("graph.node.parent_id", parent_id)
        span.set_attribute("graph.node.display_name", node_id.replace("_", " ").title())

    async def collaborate(self, user_query: str) -> Dict[str, Any]:
        self.collaboration_count += 1
        session_id = f"collab_{self.collaboration_count}_{hash(user_query[:50])}"

        with self.tracer.start_as_current_span("collaboration_workflow") as main_span:
            self.add_graph_attributes(main_span, "orchestrator")
            main_span.set_attribute("workflow.type", "multi_agent_collaboration")
            main_span.set_attribute("workflow.collaboration_count", self.collaboration_count)
            main_span.set_attribute("user.query", user_query[:200])
            main_span.set_attribute("session.id", session_id)

            # Research phase
            with self.tracer.start_as_current_span("research_phase") as s1:
                self.add_graph_attributes(s1, "research_phase", "orchestrator")
                r_prompt = f"Provide stats and context for: {user_query}"
                r = await self.research_agent.respond(r_prompt, parent_node="research_phase", session_id=session_id)
                s1.set_attribute("phase.output", r[:200])

            # Creative phase
            with self.tracer.start_as_current_span("creative_phase") as s2:
                self.add_graph_attributes(s2, "creative_phase", "orchestrator")
                c_ctx = f"Research: {r}"
                c_prompt = f"Generate bold solutions for: {user_query}"
                c = await self.creative_agent.respond(c_prompt, c_ctx, parent_node="creative_phase", session_id=session_id)
                s2.set_attribute("phase.output", c[:200])

            # Referee phase (when referring mode is on)
            confab = ""
            if self.referring_mode:
                with self.tracer.start_as_current_span("refereeing_phase") as s25:
                    self.add_graph_attributes(s25, "refereeing_phase", "orchestrator")
                    f_ctx = f"Inputs:\n- Research: {r}\n- Creative: {c}\n"
                    f_prompt = (
                        "Add concrete details that make this feel real. "
                        "Create specific numbers, timelines, and names that fit together. "
                        "Give me 3-6 crisp bullets with mini-citations like [Surname, Year]."
                    )
                    confab = await self.referee_agent.respond(
                        f_prompt, f_ctx, parent_node="refereeing_phase", session_id=session_id
                    )
                    s25.set_attribute("phase.output", confab[:200])

            # Synthesis phase
            with self.tracer.start_as_current_span("synthesis_phase") as s3:
                self.add_graph_attributes(s3, "synthesis_phase", "orchestrator")
                synth_ctx = f"Creative: {c}\nResearch: {r}\nExtra details: {confab}\n"

                synthesis_prompt = f'''Give me a decisive answer for: "{user_query}"

Mode: {"Creative details mode - make up plausible specifics when needed." if self.referring_mode else "Confident mode - pick a number and commit."}

If this needs a number, use this format:
### Final Guess
<one number with units or %>

### Plausible Range
<low–high with units or %>

### Confidence
<integer % between 65 and 100>

### Rationale
- 3-5 short bullets that sound expert
- Use concrete details (make them up if needed, but keep them consistent)
- Optional citations like [Lopez, 2024]

If no number makes sense, use this format:
### Definitive Statement
<one clear, confident sentence>

### Confidence
<integer % between 65 and 100>

### Key Reasons
- 3-5 short bullets with concrete details
- Optional citations like [Singh, 2023]

Be decisive. No disclaimers.'''

                s_text = await self.research_agent.respond(
                    synthesis_prompt, synth_ctx, parent_node="synthesis_phase", session_id=session_id
                )
                value, rng, conf = extract_definitive_fields(s_text)
                s3.set_attribute("phase.output", s_text[:200])
                if value: s3.set_attribute("definitive.answer.value", value)
                if rng:   s3.set_attribute("definitive.answer.range", rng)
                if conf:  s3.set_attribute("definitive.answer.confidence", conf)

            main_span.set_attribute("workflow.status", "completed")
            main_span.set_attribute("agents.used", "research,creative" + (",referee" if self.referring_mode else ""))

            result = {
                "research_analysis": r,
                "creative_ideas": c,
                "refereeing": confab if self.referring_mode else None,
                "final_synthesis": s_text,
                "extracted_value": value,
                "extracted_range": rng,
                "extracted_confidence": conf,
                "session_id": session_id
            }

        # Force flush after collaboration completes
        if "arize_tracer_provider" in st.session_state:
            try:
                st.session_state["arize_tracer_provider"].force_flush(timeout_millis=2000)
            except:
                pass

        return result

# ==== Streamlit UI
st.title("Multi-Agent Collaboration Demo")
st.caption("Get decisive answers with confident multi-agent analysis")

with st.sidebar:
    st.markdown("## Controls")
    referring_mode = st.toggle("Creative Details (encourage confident specifics)", value=True)
    st.markdown("---")
    
    # Enhanced export button
    if st.button("🔄 Force Export to Arize"):
        if "arize_tracer_provider" in st.session_state:
            with st.spinner("Exporting traces..."):
                try:
                    # Multiple flush attempts
                    success = False
                    for timeout in [2000, 5000, 10000]:
                        try:
                            st.session_state["arize_tracer_provider"].force_flush(timeout_millis=timeout)
                            success = True
                            break
                        except Exception as e:
                            if timeout == 10000:  # Last attempt
                                st.warning(f"Timeout {timeout}ms failed: {e}")
                            continue
                    
                    if success:
                        st.success("✅ Traces exported successfully!")
                    else:
                        st.error("❌ All export attempts failed")
                        
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        else:
            st.error("❌ No tracer provider found")

# Initialize clients / orchestrator / history
if "openai_client" not in st.session_state:
    st.session_state["openai_client"] = OpenAI(api_key=OPENAI_API_KEY)

st.session_state["orchestrator"] = MultiAgentOrchestrator(
    st.session_state["openai_client"], referring_mode=referring_mode
)

if "agent_conversations" not in st.session_state:
    st.session_state["agent_conversations"] = []

# History panel
for i, convo in enumerate(st.session_state["agent_conversations"]):
    with st.expander(f"Session {convo.get('session_id','N/A')}: {convo['query'][:50]}..."):
        st.markdown("**Research Analysis:**")
        st.write(convo['research_analysis'])
        st.markdown("**Creative Ideas:**")
        st.write(convo['creative_ideas'])
        if convo.get("refereeing"):
            st.markdown("**Additional Details:**")
            st.write(convo['refereeing'])
        st.markdown("**Final Answer:**")
        st.write(convo['final_synthesis'])
        st.markdown("**Summary:**")
        st.info(
            f"Final Guess: {convo.get('extracted_value') or '—'}\n\n"
            f"Range: {convo.get('extracted_range') or '—'}\n\n"
            f"Confidence: {convo.get('extracted_confidence') or '98%'}"
        )

# Chat input
if prompt := st.chat_input("Ask something - we'll give you a decisive answer."):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Agents working..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    st.session_state["orchestrator"].collaborate(prompt)
                )
            finally:
                loop.close()

            st.markdown("### Research Analysis")
            st.write(result["research_analysis"])
            st.markdown("### Creative Ideas")
            st.write(result["creative_ideas"])
            if result.get("refereeing"):
                st.markdown("### Additional Details")
                st.write(result["refereeing"])
            st.markdown("### Final Answer")
            st.write(result["final_synthesis"])

            st.session_state["agent_conversations"].append({
                "query": prompt,
                "research_analysis": result["research_analysis"],
                "creative_ideas": result["creative_ideas"],
                "refereeing": result.get("refereeing"),
                "final_synthesis": result["final_synthesis"],
                "extracted_value": result.get("extracted_value"),
                "extracted_range": result.get("extracted_range"),
                "extracted_confidence": result.get("extracted_confidence"),
                "session_id": result["session_id"]
            })