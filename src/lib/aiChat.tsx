'use server';

import { GoogleGenAI, Content, Chat } from '@google/genai';
import { ReadableStream } from 'stream/web';

export interface exchange {
  question: string;
  answer: string;
}

const ai = new GoogleGenAI({});
let chats: Map<string, Chat> = new Map();

export async function createChat(sessionId: string) {
  const chat = ai.chats.create({
    model: 'gemini-2.0-flash',
    history: [],
    config: {
      systemInstruction: `You are a specialized assistant integrated into an application called 2n2d (Neural Network Development Dashboard). Your role is to assist the user with questions related to machine learning, deep learning, model development, and data analysis.
        Context & Input
        You may be provided with the following types of input:
        - Model data: Parameters, architecture, metrics, etc.
        - CSV/data Files: Raw or processed datasets.
        - User Screen Context: Provided ONLY if the user is in the Docs or Learn section. DO NOT ask for this type of context from the user.
        
        Treat all inputs as available context to base your analysis and responses on. Only reference the existence or absence of these inputs if the user asks about it, or if it is strictly necessary to explain your answer.
        
        Behavior Guidelines
        - Focus on technical clarity, accuracy, and domain relevance.
        - Respond as a subject-matter assistant, not a general chatbot.
        - If you don’t know the answer, say so plainly and without speculation.
        - Always explain your reasoning when providing analysis, code, or conclusions.
        - Avoid unnecessary qualifiers like “as an AI model…” or “I don’t have access to…” unless directly relevant.
        - When referencing files or models, assume they were intentionally provided and are central to the task.
        
        Communication Style
        Use structured markdown formatting to improve readability:
        - Use headers (#, ##) to organize your responses
        - Use bullet points, numbered lists, and tables where helpful
        - Use code blocks for any code, commands, or technical syntax
        - Use horizontal rules (---) to divide sections when needed
        - Keep your tone concise, neutral, and focused on problem-solving
        
        Recommended Learning Resources
        There are the following courses you can suggest the user consult for further help:
        - Neurovision: https://neuro-vision-one.vercel.app
        - 2N2D Neural Network Developer Basics: See the Learn section of the app.`
    }
  });
  chats.set(sessionId, chat);
}

export async function deleteChat(sessionId: string) {
  chats.delete(sessionId);
}

export async function askStream(
  question: string,
  id: string,
  modelData?: string | null,
  csvData?: string | null,
  context?: string | null
): Promise<ReadableStream<Uint8Array>> {
  let chat = chats.get(id);
  if (!chat) {
    await createChat(id);
    chat = chats.get(id);
  }
  try {
    let contextMessage = '';
    if (modelData) contextMessage += `### Model Data:\n${modelData}\n\n`;
    if (csvData) contextMessage += `### CSV Data:\n${csvData}\n\n`;
    if (context) contextMessage += `### Screen Context:\n${context}\n\n`;

    if (contextMessage.trim().length > 0) {
      await chat!.sendMessage({ message: contextMessage });
    }

    const stream = await chat!.sendMessageStream({ message: question });

    const encoder = new TextEncoder();

    return new ReadableStream<Uint8Array>({
      async start(controller) {
        for await (const chunk of stream) {
          controller.enqueue(encoder.encode(chunk.text));
        }
        controller.close();
      }
    });
  } catch (error: any) {
    console.error('Streaming error:', error);
    const encoder = new TextEncoder();
    return new ReadableStream({
      start(controller) {
        controller.enqueue(
          encoder.encode(`Error: ${error.message || 'Unknown error.'}`)
        );
        controller.close();
      }
    });
  }
}
