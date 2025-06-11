"use server"

import {getSession, updateSession, Session} from "./sessionManager";
import {deleteFile} from "@/lib/fileHandler/supaStorage";

export async function updateData(id: number, data: any) {
    let session = await getSession(id);
    session.csvResult = data;
    session.csvName = data.summary.filename;
    await updateSession(session);
}

export async function updateVis(id: number, modelData: any, fileName: string) {
    let session = await getSession(id);
    session.visResult = modelData;
    session.onnxName = fileName;
    await updateSession(session);
}

export async function updateChat(id: number, chat: any) {
    let session = await getSession(id);
    session.chat = chat;
    await updateSession(session);
}

export async function updateName(id: number, newName: string) {
    let session = await getSession(id);
    session.name = newName;
    await updateSession(session);
}

export async function updateOptimize(id: number, url: string, optResult: any) {
    let session = await getSession(id);
    session.optimizedFileUrl = url;
    session.optResult = optResult;
    await updateSession(session);
}

export async function updateOnnxUrl(id: number, url: string) {
    let session = await getSession(id);
    session.onnxUrl = url;
    await updateSession(session);
}

export async function updateCsvUrl(id: number, url: string) {
    let session = await getSession(id);
    session.csvUrl = url;
    await updateSession(session);
}

export async function deleteCsv(id: number) {
    let session = await getSession(id);
    if (!session) return;
    if (!session.csvUrl || session.csvUrl.length == 0) return;

    await deleteFile("csv", session.csvUrl);
    session.csvUrl = "";
    session.csvResult = "";
    session.csvName = "";

    if (session.optimizedFileUrl && session.optimizedFileUrl.length > 0) {
        await deleteFile("rezult", session.optimizedFileUrl);
        session.optimizedFileUrl = "";
        session.optResult = "";
    }


    await updateSession(session);
}

export async function deleteOnnx(id: number) {
    let session = await getSession(id);
    if (!session) return;
    if (!session.onnxUrl || session.onnxUrl.length == 0) return;

    await deleteFile("onnx", session.onnxUrl);
    session.onnxUrl = "";
    session.onnxName = "";
    session.visResult = "";

    if (session.optimizedFileUrl && session.optimizedFileUrl.length > 0) {
        await deleteFile("rezult", session.optimizedFileUrl);
        session.optimizedFileUrl = "";
        session.optResult = "";
    }


    await updateSession(session);
}