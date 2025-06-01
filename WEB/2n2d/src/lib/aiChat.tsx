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

export async function ask(question: string, history: exchange[], modelData?: string | null, csvData?: string | null): Promise<exchange> {
    const response: exchange = {question: question, answer: ""};
    const messages: ChatCompletionMessageParam[] = [];
    messages.push({
        role: "system",
        content: "You are a useful assistant integrated in an app called 2n2d(neural network development dashboard). Your task is to help the user with his/her questions. You will be provided with the files the user is working if, if any. Try to be specialized and explain your reasoning. Style your responses with markdown, try to utilize tables, headers, spacing lines, and other markdown styling to make your answers more clear. If you don't know the answer, just say that you don't know. You might be provided with model data and csv data. Analize and help the user with any questions it has"
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
            content: "No csv data provided"
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
            content: "No model data provided"
        })
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