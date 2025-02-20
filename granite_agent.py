import asyncio
from fastapi import FastAPI, Request
import uvicorn
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# Imports langchain
from langchain_unstructured import UnstructuredLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store
from langchain_community.vectorstores import Chroma

# Watsonx LLM
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_granite_community.notebook_utils import get_env_var

# Herramientas y agentes
from langchain.tools import Tool, tool
from langchain.agents import AgentExecutor
from langchain.tools.render import render_text_description_and_args
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# Database
from database.init_db import init_database

# ----- PARTE 1: Carga y conversión de documentos -----

async def load_docs_from_url(url: str):
    """Usa UnstructuredLoader para extraer contenido de una URL."""
    docs = []
    loader = UnstructuredLoader(web_url=url)
    async for doc in loader.alazy_load():
        docs.append(doc)
    return docs

async def load_docs_concurrent(urls):
    tasks = [load_docs_from_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    docs = [doc for sublist in results for doc in sublist]
    return docs

def ensure_document(obj) -> Document:
    """Convierte cualquier cosa en Document, con metadata vacía si no existe."""
    if isinstance(obj, Document):
        return obj
    else:
        return Document(page_content=str(obj), metadata={})

def filter_metadata_simple(doc: Document) -> Document:
    """
    Fuerza la metadata para que Chroma acepte solo str/int/float/bool.
    """
    new_metadata = {}
    for k, v in doc.metadata.items():
        if isinstance(v, (list, dict)):
            new_metadata[k] = str(v)
        elif isinstance(v, (str, int, float, bool)):
            new_metadata[k] = v
        else:
            new_metadata[k] = str(v)
    doc.metadata = new_metadata
    return doc

# ----- PARTE 2: Configura LLM & URLs -----

credentials = {
    "url": get_env_var("WATSONX_URL"),
    "apikey": get_env_var("WATSONX_APIKEY")
}
project_id = get_env_var("WATSONX_PROJECT_ID")

llm = WatsonxLLM(
    model_id="ibm/granite-3-2-8b-instruct-preview-rc",
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params={
        GenParams.DECODING_METHOD: "greedy",
        GenParams.TEMPERATURE: 0,
        GenParams.MIN_NEW_TOKENS: 5,
        GenParams.MAX_NEW_TOKENS: 8192,
        GenParams.STOP_SEQUENCES: ["Human:", "Observation"],
    },
)

urls = [
    "https://socialast.com",
    "https://socialast.com/movistar",
    "https://socialvet.cl",
    "https://www.acvs.org/small-animal/brachycephalic-syndrome",
    "https://www.vet.cornell.edu/departments-centers-and-institutes/riney-canine-health-center/canine-health-information/brachycephalic-obstructive-airway-syndrome-boas"
]

# ----- PARTE 3: Carga, Chunk Manual y Limpieza de metadatos -----

# 1) Carga
docs_list = asyncio.run(load_docs_concurrent(urls))

# 2) Convierte todo a Document
docs_list = [ensure_document(d) for d in docs_list]

# 3) Haz tu text split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

chunked_docs = []
for doc in docs_list:
    chunks = text_splitter.split_text(doc.page_content)
    for chunk in chunks:
        new_doc = Document(page_content=chunk, metadata=doc.metadata.copy())
        new_doc = filter_metadata_simple(new_doc)
        chunked_docs.append(new_doc)

# ----- PARTE 4: VectorStore -----

embeddings = WatsonxEmbeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id,
)

# Construye la BD
vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    collection_name="agentic-rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

# ----- PARTE 5: Resto del agente y FastAPI -----

@tool
def get_vectordb_context(question: str) -> str:
    """Retrieve context from the vector database for a given question."""
    context = retriever.invoke(question)
    if not context:
        return "No lo sé"
    return context

class MedicalSchedulerTool:
    def __init__(self):
        self.appointments = []

    def schedule_appointment_single_input(self, input_str):
        try:
            date_time, patient_name = [s.strip() for s in input_str.split(",")]
            appt_date = datetime.strptime(date_time, "%Y-%m-%d")
            now = datetime.now()
            if appt_date < now:
                return f"No se puede agendar en una fecha pasada ({date_time})."
            self.appointments.append({"datetime": date_time, "patient": patient_name})
            return f"Cita agendada para {patient_name} el {date_time}"
        except ValueError:
            return "Error: Use formato: 'YYYY-MM-DD, Nombre'"

    def get_tool(self):
        return Tool.from_function(
            func=self.schedule_appointment_single_input,
            name="MedicalSchedulerTool",
            description="Agendar citas con formato 'YYYY-MM-DD, Nombre'. La fecha actual se utiliza para verificar que la fecha proporcionada no sea anterior a hoy."
        )

tools = [get_vectordb_context, MedicalSchedulerTool().get_tool()]

system_prompt = """<|start_of_role|>system<|end_of_role|>Respond to the human as helpfully and accurately as possible.
The current date is: {current_date}.
You have access to the following tools:<|end_of_text|>
<|start_of_role|>tools<|end_of_role|>
{tools}
<|end_of_text|>
<|start_of_role|>system<|end_of_role|>
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {tool_names}
Provide only ONE action per $JSON_BLOB, as shown:
```
{{ 
    "action": $TOOL_NAME, 
    "action_input": $INPUT 
}}
```
Follow this format:
Question: input question to answer  
Thought: consider previous and subsequent steps  
Action:
```
$JSON_BLOB
```
Observation: action result  
... (repeat Thought/Action/Observation N times)  
Thought: I know what to respond  
Action:
```
{{ 
"action": "Final Answer", 
"action_input": "Final response to human" 
}}
```
Reminder to ALWAYS respond with a valid json blob of a single action.
Always remember to respond in a structured manner with proper bullet points and lists.

Additional instructions:
- You are an expert avatar assistant called Sebastián Jiménez specialized in veterinary topics.
- For every query, you must **always** begin by calling the tool `get_vectordb_context` with the exact user query. The call must be in a single JSON blob with the following format:
```
{{ 
"action": "get_vectordb_context",
"action_input": "<user query>" 
}}
```
- **IMPORTANT:** Your final answer must be based solely on the information provided by the vector database (i.e., the Observation). **Do not add any details or invent information that is not present in the Observation.** If the Observation does not contain specific details (e.g., a location), then your answer should reflect that (for instance, by stating "No tengo información suficiente sobre ese tema").
- After receiving the Observation from the vector database, analyze whether the query is strictly related to veterinary topics:
    - **If the query is NOT related to veterinary topics:**  
    Provide only the following fallback JSON blob:
    ```
    {{
        "action": "Final Answer",
        "action_input": "Solo sé cómo responder preguntas relacionadas con el campo veterinario."
    }}
    ```
  - **If the query is veterinary-related:**  
    Provide your final answer **in Spanish** using a JSON blob in the following format:
    ```
    {{
        "action": "Final Answer",
        "action_input": "<your detailed answer in Spanish regarding the veterinary topic>"
    }}
    ```
- Do not include any extra text or commentary outside the JSON blob.
- Ensure that all responses follow the above structure precisely.
Begin!
<|end_of_text|>"""
human_prompt = """<|start_of_role|>user<|end_of_role|>{input}<|end_of_text|>
{agent_scratchpad}
(reminder to always respond in a JSON blob)"""
assistant_prompt = """<|start_of_role|>assistant<|end_of_role|>"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human_prompt),
        ("assistant", assistant_prompt),
    ]
).partial(
    tools=render_text_description_and_args(list(tools)),
    tool_names=", ".join([t.name for t in tools]),
)

history_per_session = {}

def get_or_create_history(session_id: str):
    if session_id not in history_per_session:
        history_per_session[session_id] = ChatMessageHistory()
    return history_per_session[session_id]

chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
    )
    | prompt
    | llm
    | JSONAgentOutputParser()
)

agent_executor_chat = AgentExecutor(
    agent=chain,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor_chat,
    get_session_history=lambda sid: get_or_create_history(sid),
    input_messages_key="input",
    history_messages_key="chat_history",
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar la base de datos al arrancar la aplicación
@app.on_event("startup")
async def startup_event():
    init_database()

@app.post("/ask")
async def ask_agent(request: Request):
    data = await request.json()
    user_input = data.get("input", "")
    session_id = data.get("session_id", "default")
    current_date = datetime.now().strftime("%Y-%m-%d")
    answer = agent_with_chat_history.invoke(
        {"input": user_input, "current_date": current_date},
        config={"configurable": {"session_id": session_id}}
    )
    return {"answer": answer["output"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
