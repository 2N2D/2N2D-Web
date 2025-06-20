"use client"
import React, {useEffect} from "react"
import * as ort from "onnxruntime-web"
import {downloadFile} from "@/lib/fileHandler/supaStorage";
import {getSession, Session} from "@/lib/sessionHandling/sessionManager"

export default function ModelTest() {
    function mapInput(session: ort.InferenceSession) {
        const inputs: Record<string, { type: string, dimensions: (number | string)[] }> = {};
        for (const meta of session.inputMetadata) {
            if (!meta.isTensor) return;
            inputs[meta.name] = {
                type: meta.type,

            };
        }

        return inputs;
    }

    async function loadData() {
        const idStr = sessionStorage.getItem("currentSessionId");
        if (!idStr)
            return;
        const id = parseInt(idStr);

        const currentSession: Session = await getSession(id);
        if (!currentSession)
            return;

        if (!currentSession.onnxUrl)
            return;

        const onnxBlob = await downloadFile("onnx", currentSession.onnxUrl);
        if (!onnxBlob || typeof onnxBlob === "string")
            return;

        const buffer = await onnxBlob?.arrayBuffer();
        const session = await ort.InferenceSession.create(buffer);

        console.log(JSON.stringify(mapInput(session)))
    }

    useEffect(() => {
        loadData()
    }, []);

    return <div>
        Test
    </div>
}