import { NextRequest } from 'next/server';
import { askStream } from '@/lib/aiChat';

export async function POST(req: NextRequest) {
  const { question, sessionId, modelData, csvData, context } = await req.json();

  const stream = await askStream(
    question,
    sessionId,
    modelData,
    csvData,
    context
  );

  return new Response(
    new ReadableStream({
      async start(controller) {
        for await (const chunk of stream) {
          controller.enqueue(chunk);
        }
        controller.close();
      }
    }),
    {
      headers: {
        'Content-Type': 'text/plain',
        'Cache-Control': 'no-cache'
      }
    }
  );
}
