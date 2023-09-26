import "dotenv/config";
import weaviate from "weaviate-ts-client";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PromptTemplate } from "langchain/prompts";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";
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

const fastestModel = new ChatOpenAI({
  model: "gpt-3.5-turbo",
});

const template = `Dada la conversación y la pregunta de seguimiento, actúe como un veterinario para responder sobre la salud y comportamiento de mascotas.
Historia del Chat:
{chat_history}
Pregunta de seguimiento: {question}
Su respuesta debe seguir el siguiente formato:
\`\`\`
Utilice el siguiente contexto para responder la pregunta.
Si no sabe la respuesta, no trate de adivinar, solamente diga que desconoce la información.
----------------
<Historia de chat relevante y contexto>
\`\`\`
Respuesta:`;

const chain = ConversationalRetrievalQAChain.fromLLM(
  model,
  vectorStore.asRetriever(),
  {
    returnSourceDocuments: true,
    questionGeneratorChainOptions: {
      template,
      llm: fastestModel
    },
    memory: new BufferMemory({
      memoryKey: "chat_history", // Must be set to "chat_history"
      inputKey: "question",
      outputKey: "text",
    }),
  }
);

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
