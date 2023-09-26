import "dotenv/config";
import weaviate from "weaviate-ts-client";
import { WeaviateStore } from "langchain/vectorstores/weaviate";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { YoutubeLoader } from "langchain/document_loaders/web/youtube";

// Create Vector Database Client
const client = weaviate.client({
  scheme: process.env.WEAVIATE_SCHEME,
  host: process.env.WEAVIATE_HOST,
  apiKey: new weaviate.ApiKey(process.env.WEAVIATE_API_KEY)
});

// Define list of videos to load to the database
const videos = [
  "https://youtube.com/watch?v=l2293pac4dU",
  "https://youtube.com/watch?v=6oJbSz74_Cw",
  "https://youtube.com/watch?v=hHCTjCwnSG4",
];

// Get transcripts as documents
const loaders = videos.map((url) =>
  YoutubeLoader.createFromUrl(url, {
    language: "es",
    addVideoInfo: true,
  })
);
const docs = await Promise.all(loaders.map((loader) => loader.load()));

// Split documents into chunks of 1000 characters
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 0,
});
const texts = await splitter.splitDocuments(docs.flat());

// Load documents into Vector Database 
await WeaviateStore.fromDocuments(texts, new OpenAIEmbeddings(), {
  client,
  indexName: "Veterinario",
  textKey: "text",
  metadataKeys: ["title", "description", "source"],
});
console.log(`${videos.length} videos loaded into Weaviate.`);