import "dotenv/config";
import weaviate from "weaviate-ts-client";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PromptTemplate } from "langchain/prompts";
import { RetrievalQAChain, loadQAStuffChain } from "langchain/chains";
import { WeaviateStore } from "langchain/vectorstores/weaviate";
import { input } from "@inquirer/prompts";

// Connect to Vector Database
const client = weaviate.client({
  scheme: process.env.WEAVIATE_SCHEME,
  host: process.env.WEAVIATE_HOST,
  apiKey: new weaviate.ApiKey(process.env.WEAVIATE_API_KEY),
});

const vectorStore = await WeaviateStore.fromExistingIndex(
  new OpenAIEmbeddings(),
  {
    client,
    indexName: "Veterinario",
    textKey: "text",
    metadataKeys: ["title", "description", "source"],
  }
);
const retriever = vectorStore.asRetriever();

// Load Chat Model and Prompt
const model = new ChatOpenAI({
  model: "gpt-4",
});

const template = `Actúe como un Veterinario, utilice el siguiente context para responder preguntas sobre salud y comportamiento de mascotas.
Siempre consteste al final con "Muchas gracias por tu pregunta, ¿te puedo ayudar en algo más?" - Utilice lenguaje natural como si estuviera en una conversación real.

{context}

Pregunta: {question}
Respuesta:`;

const QA_CHAIN_PROMPT = new PromptTemplate({
  inputVariables: ["context", "question"],
  template,
});

// Start QA Chain
const chain = new RetrievalQAChain({
  combineDocumentsChain: loadQAStuffChain(model, { prompt: QA_CHAIN_PROMPT }),
  retriever,
  returnSourceDocuments: true,
  inputKey: "question",
});

// Application Loop
async function run() {
  const question = await input({ message: "Pregunta:" });
  if (
    question === "exit" ||
    question === "quit" ||
    question === "q" ||
    question === "salir"
  ) {
    process.exit(0);
  }

  const query = await chain.call({ question });
  console.log(query.text);
  const sources = [
    ...new Set(query.sourceDocuments.map((doc) => doc.metadata.source)),
  ];
  console.log("\nVideos relacionados:");
  sources.forEach((videoId) => {
    console.log(`📺 https://youtube.com/watch?v=${videoId}`);
  });
  await run();
}
await run();
