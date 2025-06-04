"use client"
import React, {useState, useEffect, FormEvent} from "react"
import {requestOptimized, startOptimization} from "@/lib/2n2dAPI";
import {getSessionTokenHash} from "@/lib/auth/authentication";
import "./styles.css"
import ONNXUploader from "@/components/fileUploadElements/ONNXUploader";
import CSVUploader from "@/components/fileUploadElements/CSVUploader";

function Optimize() {
    const [features, setFeatures] = useState<string[]>([]);
    const [status, setStatus] = useState<string>("");
    const [progress, setProgress] = useState<number>(-1);
    const [csvFileName, setCsvFileName] = useState<string>("");
    const [onnxFileName, setOnnxFileName] = useState<string>("");
    const [result, setResult] = useState<any>(null);

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
        setFeatures([]);

        if (sessionStorage.getItem("modelName"))
            setOnnxFileName(sessionStorage.getItem("modelName")!);

        if (!sessionStorage.getItem("csvData") || !sessionStorage.getItem("modelData")) {
            return;
        }

        let csv = JSON.parse(sessionStorage.getItem("csvData")!);

        setCsvFileName(csv.summary.filename);


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

        setProgress(0);
        setStatus("Starting optimization...");

        const _result = await startOptimization(Ifeatures, target, maxEpochs);
        setResult(_result);
        console.log(result);
    }

    async function downloadOptimized() {
        const file = await requestOptimized();

        if (file)
            console.log(file);
    }

    return (
        <main className={"pageOpt"}>
            <div className={"flex  w-full gap-[0.1rem]"}>
                <div className={"flex flex-col"} style={{width: "100%"}}>
                    <form className={"optimizeForm area"} style={{width: "100%"}} onSubmit={optimize}>
                        <h2 className={"subtitle"}>Optimization settings</h2>
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

                        <input type="submit" id="opt-start-optimization" value="Start Optimization"
                               disabled={progress > -1 && progress < 100 ? true : false}/>
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

                    </div>
                </div>
                <div>
                    <div className={"dataArea area vertical"}>
                        <h1><b>ONNX file:</b> {onnxFileName}</h1>
                        <div className={"flex gap-[1rem]"}>
                            <ONNXUploader callBack={populateLists}/>
                            <button className={"deleteButton"} onClick={() => {
                                sessionStorage.removeItem("modelData");
                                populateLists()
                            }}>
                                Clear Data <i className="fa-solid fa-trash-xmark"></i>
                            </button>
                        </div>
                    </div>
                    <div className={"dataArea area vertical"}>
                        <h1><b>CSV file:</b> {csvFileName}</h1>
                        <div className={"flex gap-[1rem]"}>
                            <CSVUploader callBack={populateLists}/>
                            <button className={"deleteButton"} onClick={() => {
                                sessionStorage.removeItem("csvData");
                                populateLists()
                            }}>
                                Clear Data <i className="fa-solid fa-trash-xmark"></i>
                            </button>
                        </div>
                    </div>
                </div>
                <div className={"area vertical title"}>
                    <h1>Optimization</h1>
                </div>
            </div>

            <div className={`resultArea ${progress == 100 ? "area" : ""}`}
                 style={progress == 100 ? {height: "100%"} : {height: 0}}>
                <h1 className={"main subtitle"}>Optimization results:</h1>
                <div className={"flex flex-col gap-[1rem] p-[1rem]"}>
                    <h2 className={"subtitle"}>Best configuration:</h2>
                    <div className={"result"}>
                        <div className={"info"}><h2>Neurons:</h2> {result?.best_config.neurons}</div>
                        <div className={"info"}><h2>Layers:</h2> {result?.best_config.layers}</div>
                        <div className={"info"}><h2>Test loss:</h2> {result?.best_config.test_loss}</div>
                        <div className={"info"}><h2>R2 score:</h2> {result?.best_config.r2_score}</div>
                    </div>
                </div>
                <table className={"table"}>
                    <thead>
                    <tr>
                        <th>Configuration</th>
                        <th>Neurons</th>
                        <th>Layers</th>
                        <th>Test loss:</th>
                        <th>R2 score:</th>
                    </tr>
                    <tr>
                        <td>Best config</td>
                        <td>{result?.best_config.neurons}</td>
                        <td>{result?.best_config.layers}</td>
                        <td>{result?.best_config.test_loss}</td>
                        <td>{result?.best_config.r2_score}</td>
                    </tr>
                    </thead>
                    <tbody>
                    {result?.results.map((res: {
                        neurons: string | number | bigint | boolean | React.ReactElement<unknown, string | React.JSXElementConstructor<any>> | Iterable<React.ReactNode> | React.ReactPortal | Promise<string | number | bigint | boolean | React.ReactPortal | React.ReactElement<unknown, string | React.JSXElementConstructor<any>> | Iterable<React.ReactNode> | null | undefined> | null | undefined;
                        layers: string | number | bigint | boolean | React.ReactElement<unknown, string | React.JSXElementConstructor<any>> | Iterable<React.ReactNode> | React.ReactPortal | Promise<string | number | bigint | boolean | React.ReactPortal | React.ReactElement<unknown, string | React.JSXElementConstructor<any>> | Iterable<React.ReactNode> | null | undefined> | null | undefined;
                        test_loss: string | number | bigint | boolean | React.ReactElement<unknown, string | React.JSXElementConstructor<any>> | Iterable<React.ReactNode> | React.ReactPortal | Promise<string | number | bigint | boolean | React.ReactPortal | React.ReactElement<unknown, string | React.JSXElementConstructor<any>> | Iterable<React.ReactNode> | null | undefined> | null | undefined;
                        r2_score: string | number | bigint | boolean | React.ReactElement<unknown, string | React.JSXElementConstructor<any>> | Iterable<React.ReactNode> | React.ReactPortal | Promise<string | number | bigint | boolean | React.ReactPortal | React.ReactElement<unknown, string | React.JSXElementConstructor<any>> | Iterable<React.ReactNode> | null | undefined> | null | undefined;
                    }, i: number) =>
                        <tr key={i}>
                            <td>Config {i + 1}</td>
                            <td>{res.neurons}</td>
                            <td>{res.layers}</td>
                            <td>{res.test_loss}</td>
                            <td>{res.r2_score}</td>
                        </tr>)}
                    </tbody>
                </table>

                <button onClick={downloadOptimized}>Download optimized <i
                    className="fa-solid fa-file-arrow-down"></i></button>
            </div>

        </main>
    )
}

export default Optimize;