import {getSessionTokenHash} from "@/lib/auth/authentication";

export async function startOptimization(
    selectedInputs: String[],
    targetFeature: String,
    epochs: number,
) {
    if (!targetFeature) {
        return;
    }

    try {
        return await fetch("http://127.0.0.1:8000/optimize", {
            method: "POST",
            headers: {
                "content-type": "application/json",
                "session-id": `${await getSessionTokenHash()}`
            },
            body: JSON.stringify({
                input_features: selectedInputs,
                target_feature: targetFeature,
                max_epochs: epochs,
            }),
        });
    } catch (error) {
        console.error("Error during optimization:", error);
        throw error;
    }
}

export async function sendCSV(formData: any) {
    try {
        const result = await fetch("http://127.0.0.1:8000/upload-csv", {
            headers: {"session-id": `${await getSessionTokenHash()}`},
            method: "POST",
            body: formData,
        });

        const response = await result.json();
        sessionStorage.setItem("csvData", JSON.stringify(response));
        return response;
    } catch (error) {
        console.error("Upload failed", error);
    }
}

export async function sendModel(formData: any) {
    try {
        const result = await fetch("http://127.0.0.1:8000/upload-model", {
            headers: {"session-id": `${await getSessionTokenHash()}`},
            method: "POST",
            body: formData,
        });

        const response = await result.json();
        sessionStorage.setItem("modelData", JSON.stringify(response));
        return response;
    } catch (error) {
        console.error("Upload failed", error);
    }
}

export async function requestOptimized() {
    try {
        const res = await fetch("http://127.0.0.1:8000/download-optimized", {
            headers: {"session-id": `${await getSessionTokenHash()}`},
            method: "GET",
        });

        if (!res.ok) {
            throw new Error("Failed to fetch file");
        }

        const data = await res.json();
        const {base64, filename} = data;

        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }

        const blob = new Blob([bytes], {type: "application/octet-stream"});

        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = filename || "download.onnx";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        return res.json;
    } catch (error) {
        console.error("Download failed", error);
    }
}