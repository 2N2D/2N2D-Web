import { NextRequest } from 'next/server';
import { askStream } from '@/lib/aiChat';

export const runtime = 'nodejs'; // Ensure Node.js for streaming

export async function POST(req: NextRequest) {
  const { question, sessionId, modelData, csvData, context } = await req.json();

  const stream = await askStream(
    question,
    sessionId,
    modelData,
    csvData,
    context
  );

  const reader = stream.getReader();
  const readableStream = new ReadableStream({
    start(controller) {
      function push() {
        reader.read().then(({ done, value }) => {
          if (done) {
            controller.close();
            return;
          }
          controller.enqueue(value);
          push();
        });
      }
      push();
    }
  });

  return new Response(readableStream, {
    headers: {
      'Content-Type': 'application/octet-stream',
      'Transfer-Encoding': 'chunked'
    }
  });
}
