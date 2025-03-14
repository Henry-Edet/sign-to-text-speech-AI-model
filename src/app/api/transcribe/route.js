export async function POST(request) {
    const { audio } = await request.json();
  
    // Call your AI model here (e.g., using an SDK or HTTP request)
    const transcript = await callYourAIModel(audio);
  
    return Response.json({ transcript });
  }