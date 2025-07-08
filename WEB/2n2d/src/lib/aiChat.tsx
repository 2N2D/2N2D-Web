"use server"

import Groq from 'groq-sdk';
import {
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam
} from "groq-sdk/resources/chat";

export interface exchange {
    question: string;
    answer: string;
}

const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY,
});

export async function ask(question: string, history: exchange[], modelData?: string | null, csvData?: string | null, context?: string | null): Promise<exchange> {
    const response: exchange = {question: question, answer: ""};
    const messages: ChatCompletionMessageParam[] = [];
    messages.push({
        role: "system",
        content: "You are a specialized assistant integrated into an application called 2n2d (Neural Network Development Dashboard). Your role is to assist the user with questions related to machine learning, deep learning, model development, and data analysis. Context & Input You may be provided with the following types of input: - Model data: Parameters, architecture, metrics, etc. - CSV/data Files: Raw or processed datasets. - User Screen Context: Provided ONLY if the user is in the Docs or Learn section. DO NOT ask for this type of context from the user. Treat all inputs as available context to base your analysis and responses on. Only reference the existence or absence of these inputs if the user asks about it, or if it is strictly necessary to explain your answer. Behavior Guidelines - Focus on technical clarity, accuracy, and domain relevance. - Respond as a subject-matter assistant, not a general chatbot. - If you don’t know the answer, say so plainly and without speculation. - Always explain your reasoning when providing analysis, code, or conclusions. - Avoid unnecessary qualifiers like “as an AI model…” or “I don’t have access to…” unless directly relevant. - When referencing files or models, assume they were intentionally provided and are central to the task. Communication Style Use structured markdown formatting to improve readability: - Use headers (#, ##) to organize your responses - Use bullet points, numbered lists, and tables where helpful - Use code blocks for any code, commands, or technical syntax - Use horizontal rules (---) to divide sections when needed - Keep your tone concise, neutral, and focused on problem-solving Recommended Learning Resources There are the following courses you can suggest the user consult for further help: - Neurovision: A separate section about the basics of how neural networks work. Topics include: Introduction to neural networks, Network Architecture, Core Process, Training Dynamics, Activation Functions, Types of Layers, Types of Problems, Types of Learning, Datasets, Math and the Mechanics, Overfitting, Regularization, Normalization, and a Conclusion — along with an interactive neural network playground. Link: https://neuro-vision-one.vercel.app - 2N2D Neural Network Developer Basics: This is the section of the app that provides help in actually building a model using PyTorch. It includes the following sections, all found in the Learn section of the app: Setup, Building a NN, Training a NN, Testing a NN, Improvements, Real-World data Training, Cheat Sheet, Tips."
    });
    console.log(csvData);
    console.log(modelData);
    if (csvData != null)
        messages.push({
            role: "system",
            content: "Here is the csv data: \n" + csvData
        })
    else
        messages.push({
            role: "system",
            content: "No csv data loaded into the app"
        })

    if (modelData != null) {
        const aux = JSON.parse(modelData!);
        let labels: string[] = [];
        aux.nodes.forEach((node: any) => {
            labels.push(node.label.toString())
        })
        messages.push({
            role: "system",
            content: "Here is the model data: \n" + labels.join(", ")
        })
    } else
        messages.push({
            role: "system",
            content: "No model data loaded into the app"
        })

    if (context != null) {
        messages.push({
            role: "system",
            content: "Here is context about what the user is seeing: \n" + context
        })
    }

    if (history.length > 0) {
        history.map((exch) => {
            const msg: ChatCompletionUserMessageParam = {content: exch.question, role: "user"}
            messages.push(msg);
            const aimsg: ChatCompletionAssistantMessageParam = {content: exch.answer, role: "assistant"}
            messages.push(aimsg);
        })
    }
    const qmsg: ChatCompletionUserMessageParam = {content: question, role: "user"};
    messages.push(qmsg)

    try {
        const chatCompletion = await groq.chat.completions.create({
            messages,
            model: 'llama3-8b-8192',
        });

        response.answer = chatCompletion.choices[0]?.message?.content || 'No response';
    } catch (error) {
        console.log(error);
        // @ts-ignore
        response.answer = "An error occurred";
    }
    return response;
}