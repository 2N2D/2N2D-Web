"use client"
import React, {useState, useEffect, FormEvent} from "react"
import {requestOptimized, startOptimization} from "@/lib/2n2dAPI";
import {getSessionTokenHash} from "@/lib/auth/authentication";
import "./styles.css"

function Optimize() {
    const [features, setFeatures] = useState<string[]>([]);
    const [status, setStatus] = useState<string>("");
    const [progress, setProgress] = useState<number>(-1);
    // const [modelData, setModelData] = useState<any>(null);
    // const [csvData, setCsvData] = useState<any>(null);

    async function statusUpdate() {
        const eventSource = new EventSource(`http://localhost:8000/optimization-status/${await getSessionTokenHash()}`);

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setStatus(data.status);
            setProgress(data.progress);
            console.log("Progress update:", data);
        };

        eventSource.onerror = (err) => {
            console.error("SSE error:", err);
            eventSource.close();
        };

        return () => {
            eventSource.close();
        };
    }

    function populateLists() {
        if (!sessionStorage.getItem("csvData") || !sessionStorage.getItem("modelData")) {
            return;
        }

        let csv = JSON.parse(sessionStorage.getItem("csvData")!);
        let model = JSON.parse(sessionStorage.getItem("modelData")!);

        console.log(csv);
        console.log(model);

        // setModelData(model);
        // setCsvData(csv);

        let src = csv!.data;
        let feat = [];
        for (let key in src[0])
            feat.push(key);

        setFeatures(feat);
    }

    useEffect(() => {
        statusUpdate()
        populateLists();
    }, [])

    async function optimize(e: FormEvent<HTMLFormElement>) {
        e.preventDefault();
        const formData = new FormData(e.currentTarget);
        const featuresaux = formData.getAll("sIFeatures[]");
        let Ifeatures: String[] = [];
        featuresaux.forEach(feat => {
            Ifeatures.push(feat.toString());
        })
        const target = formData.get("target")?.toString()!;
        const maxEpochs = Number(formData.get("epochs"));

        const result = await startOptimization(Ifeatures, target, maxEpochs);
        console.log(result);
    }

    async function downloadOptimized() {
        const file = await requestOptimized();

        if (file)
            console.log(file);
    }

    return (
        <main className={"page"}>
            <div>
                <form className={"optimizeForm area"} onSubmit={optimize}>
                    <div className={"formGroup"}>
                        <div className={"element"}>
                            <label>Input Features:</label>
                            <ul className={"featuresList"}>
                                {
                                    features.length == 0 ?
                                        <li>Upload a CSV dataset to see available features</li> :
                                        features.map((feat, i) =>
                                            <li key={i}>
                                                <input type="checkbox" value={feat}
                                                       name={"sIFeatures[]"}/>
                                                <p>{feat}</p>
                                            </li>
                                        )
                                }
                            </ul>
                        </div>
                        <div className={"element"}>
                            <label>Target Feature:</label>
                            <select className={"targetFeature"} name={"target"}>
                                {
                                    features.length == 0 ?
                                        <option disabled>Upload a CSV dataset to see available features</option> :
                                        features.map((feat, i) => <option key={i} value={feat}>{feat}</option>)
                                }
                            </select>
                        </div>
                        <div className={"element"}>
                            <label>Maximum Epochs Per
                                Configuration:</label>
                            <input type="number" name={"epochs"} defaultValue="10"/>
                        </div>
                    </div>

                    <input type="submit" id="opt-start-optimization" value="Start Optimization"/>
                </form>
                <div style={progress != -1 ? {width: "100%"} : {width: 0}}
                     className={`progressZone ${progress != -1 ? "area" : ""}`}>
                    <h1>{status}</h1>
                    {/*<div className={progress == 100 || progress == -1 ? "hidden" : "loaderWrapper"}>*/}
                    {/*    <span className="loader"></span>*/}
                    {/*</div>*/}
                    <div className={"barWrapper"}>
                        <p>{progress}%</p>
                        <div className={"loadingBar"}>
                            <div className={`barFiller`} style={{width: `${progress}%`}}></div>
                        </div>
                    </div>
                    {
                        progress == 100 ?
                            <button onClick={downloadOptimized}>Download optimized</button> : ""
                    }
                </div>


            </div>
        </main>
    )
}

export default Optimize;